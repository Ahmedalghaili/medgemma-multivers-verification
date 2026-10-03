#!/usr/bin/env python3
"""Compare TF-IDF and biomedical PubMedBERT retrieval on HealthVer gold docs."""
import argparse, json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def read(p): return [json.loads(x) for x in Path(p).open(encoding="utf-8")]

def pool(h, mask):
    m = mask.unsqueeze(-1).expand(h.size()).float()
    return (h*m).sum(1) / m.sum(1).clamp(min=1e-9)

def encode(texts, tok, model, device):
    out=[]
    with torch.inference_mode():
        for i in range(0, len(texts), 16):
            b=tok(texts[i:i+16], padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(device)
            out.append(torch.nn.functional.normalize(pool(model(**b).last_hidden_state,
                                                            b["attention_mask"]), dim=1).cpu())
    return torch.cat(out)

def score(retrieved, gold, k):
    hits = sum(bool(set(retrieved[i][:k]) & gold[i]) for i in range(len(gold)))
    return hits / len(gold)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--annotations", default="results/pipeline_eval_100/gold_annotations.jsonl")
    ap.add_argument("--corpus", default="results/pipeline_eval_100/corpus.jsonl")
    ap.add_argument("--model", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    ap.add_argument("--output", default="results/pipeline_eval_100/retrieval_metrics.json")
    a=ap.parse_args(); ann=read(a.annotations); corpus=read(a.corpus)
    docs=[x["title"]+" "+" ".join(x["abstract"]) for x in corpus]; ids=[int(x["doc_id"]) for x in corpus]
    claims=[x["extracted_claim"] for x in ann]
    gold=[{int(e["doc_id"]) for e in x.get("gold_evidence", [])} for x in ann]
    tf=TfidfVectorizer(stop_words="english"); dm=tf.fit_transform(docs)
    tf_rank=[ [ids[j] for j in linear_kernel(tf.transform([q]),dm).ravel().argsort()[::-1]] for q in claims]
    device="cuda" if torch.cuda.is_available() else "cpu"
    tok=AutoTokenizer.from_pretrained(a.model); model=AutoModel.from_pretrained(a.model).to(device).eval()
    q=encode(claims,tok,model,device); d=encode(docs,tok,model,device); sim=q@d.T
    sem_rank=[[ids[j] for j in sim[i].argsort(descending=True).tolist()] for i in range(len(claims))]
    result={"n_claims":len(claims),"model":a.model,"metrics":{}}
    for name,ranks in [("tfidf",tf_rank),("semantic",sem_rank)]:
        result["metrics"][name]={f"recall_at_{k}":score(ranks,gold,k) for k in [1,3,5,10]}
    Path(a.output).write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,indent=2))

if __name__ == "__main__": main()
