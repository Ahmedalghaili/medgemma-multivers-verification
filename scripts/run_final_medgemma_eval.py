#!/usr/bin/env python3
"""Run the final frozen-verifier MedGemma evaluation.

This script is resumable. It generates real MedGemma answers and claims, then
retrieves HealthVer evidence with a biomedical semantic model and runs the
existing fine-tuned MultiVerS checkpoint. Human labels are deliberately not
invented; the produced CSV is the annotation file for final metrics.
"""
import argparse, csv, json, os, re, subprocess, sys
from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
sys.path.insert(0, str(Path(__file__).resolve().parent))
from medgemma_multivers_pipeline import semantic_scores, generate, parse_json


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).open(encoding="utf-8") if x.strip()]

def write_jsonl(rows, path):
    with Path(path).open("w", encoding="utf-8") as f:
        for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")

def answer_prompt(question):
    return f"""You are a cautious medical information assistant. Answer the question clearly.
Do not invent citations or claim certainty when evidence is limited. Distinguish established
findings from associations, and advise the reader to consult a qualified clinician for
personal diagnosis or treatment decisions.

Question: {question}

Return only the answer text."""

def claim_prompt(answer, question=""):
    return f"""Extract the atomic, medically meaningful factual claims from the answer below.
Preserve all qualifiers exactly in meaning, including may, might, can, could, associated
with, evidence suggests, appears to, and evidence is limited. Never strengthen a claim,
add a fact, or turn an association into causation. Split compound claims. Do not extract
advice, disclaimers, greetings, or questions.
Every claim must be self-contained: name the disease, drug, or population explicitly
(write "COVID-19 can cause fever", not "Fever may be present"). Extract at most 6 claims,
most important first.
Return only valid JSON on one line: {{"claims": ["claim one", "claim two"]}}

Question: {question}

Answer:
{answer}"""

def parse_claims(text):
    """Parse MedGemma's claim JSON, tolerating code fences and truncation."""
    text=re.sub(r"```(?:json)?","",text)
    try:
        obj=parse_json(text); items=obj.get("claims",[]) if isinstance(obj,dict) else obj
    except Exception:
        # Truncated or malformed JSON: keep every complete quoted claim.
        items=re.findall(r'"claim"\s*:\s*"((?:[^"\\]|\\.)*)"',text) or \
              re.findall(r'"((?:[^"\\]|\\.){15,})"',text)
    claims=[]
    for item in items:
        c=(item.get("claim","") if isinstance(item,dict) else str(item)).strip()
        if c and c.lower()!="claims": claims.append(c)
    return claims

def aggregate(docs):
    # label_probs order in MultiVerS is CONTRADICT, NEI, SUPPORT.
    choices=[]
    for doc_id,item in docs.items():
        probs=item.get("label_probs",[]); r=item.get("rationale_probs",[])
        if len(probs)!=3: continue
        rc=max(r,default=0.0)
        for label,idx in (("SUPPORT",2),("CONTRADICT",0)):
            choices.append((float(probs[idx])*max(float(rc),1e-6),label,doc_id))
    if not choices: return "NEI",0.0,None
    score,label,doc=max(choices)
    if score < .20: return "NEI",score,None
    return label,score,doc

def fallback_claims(answer):
    """Safe deterministic fallback when constrained JSON extraction fails."""
    lines=[]
    for raw in re.split(r"\n+|(?<=[.!?])\s+", answer):
        s=re.sub(r"^\s*[-*•]\s+", "", raw).strip()
        s=re.sub(r"^\s*\d+[.)]\s+", "", s).strip()
        if not s or len(s.split()) < 4: continue
        low=s.lower()
        if any(x in low for x in ["consult", "seek medical", "healthcare professional", "qualified clinician", "diagnosis and treatment decisions", "important to"]):
            continue
        if s.endswith(":"): continue
        lines.append(s if s[-1] in ".!?" else s+".")
    return lines

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--questions",default="data/medgemma_eval/questions.jsonl")
    ap.add_argument("--corpus",default="data/healthver/corpus.jsonl")
    ap.add_argument("--output_dir",default="results/medgemma_eval")
    ap.add_argument("--model_id",default="google/medgemma-4b-it")
    ap.add_argument("--checkpoint",default="multivers/finetune_results/healthver_finetuned_resume/checkpoint/epoch=3-step=2648.ckpt")
    ap.add_argument("--semantic_model",default="pritamdeka/S-PubMedBert-MS-MARCO")
    ap.add_argument("--top_k_docs",type=int,default=10)
    ap.add_argument("--max_new_tokens",type=int,default=512)
    ap.add_argument("--claim_max_new_tokens",type=int,default=512)
    ap.add_argument("--safe_fallback_only",action="store_true")
    ap.add_argument("--limit",type=int,default=0)
    args=ap.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    questions=read_jsonl(args.questions); questions=questions[:args.limit] if args.limit else questions

    answer_path=out/"generated_answers.jsonl"
    answers={x["id"]:x for x in read_jsonl(answer_path)} if answer_path.exists() else {}
    if len(answers) < len(questions):
        processor=AutoProcessor.from_pretrained(args.model_id)
        model=AutoModelForImageTextToText.from_pretrained(args.model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None).eval()
        with answer_path.open("a",encoding="utf-8") as f:
            for q in questions:
                if q["id"] in answers: continue
                prompt=answer_prompt(q["question"])
                answer=generate(processor,model,prompt,args.max_new_tokens)
                row={"id":q["id"],"question":q["question"],"prompt":prompt,
                     "generated_answer":answer,"model_name":args.model_id,
                     "generation_parameters":{"max_new_tokens":args.max_new_tokens,"do_sample":False}}
                f.write(json.dumps(row,ensure_ascii=False)+"\n"); f.flush(); answers[q["id"]]=row
        del model,processor
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    claims_path=out/"extracted_claims.jsonl"
    claims=read_jsonl(claims_path) if claims_path.exists() else []
    done={x["question_id"] for x in claims}
    # Re-load MedGemma only if claims are not complete; normally this branch is
    # reached in the same invocation, so keep a lightweight second model load.
    if len(done) < len(questions) and not args.safe_fallback_only:
        processor=AutoProcessor.from_pretrained(args.model_id)
        model=AutoModelForImageTextToText.from_pretrained(args.model_id,
          torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
          device_map="auto" if torch.cuda.is_available() else None).eval()
        with claims_path.open("a",encoding="utf-8") as f:
            for q in questions:
                if q["id"] in done: continue
                raw=generate(processor,model,claim_prompt(answers[q["id"]]["generated_answer"],q["question"]),args.claim_max_new_tokens)
                for i,claim in enumerate(parse_claims(raw),1):
                    row={"question_id":q["id"],"claim_id":f"{q['id']}-{i}","claim":claim,"extraction_method":"medgemma_json"}
                    f.write(json.dumps(row,ensure_ascii=False)+"\n"); claims.append(row)
                f.flush(); done.add(q["id"])
        del model,processor
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Some long answers occasionally fail the constrained JSON response. Keep
    # the real answer, but use a conservative sentence/bullet fallback rather
    # than silently dropping that question from the final benchmark.
    present={x["question_id"] for x in claims}
    for q in questions:
        if q["id"] in present: continue
        for i,claim in enumerate(fallback_claims(answers[q["id"]]["generated_answer"]),1):
            claims.append({"question_id":q["id"],"claim_id":f"{q['id']}-fallback-{i}","claim":claim,"extraction_method":"safe_sentence_fallback"})
    # Remove advice/disclaimer claims accidentally emitted by the model.
    claims=[x for x in claims if not any(t in x["claim"].lower() for t in ["consult a qualified", "consult a healthcare", "seek medical attention", "diagnosis and treatment decisions"])]
    # Keep the benchmark balanced and tractable: at most four atomic claims
    # per answer, preserving the model's order.
    limited=[]; counts={}
    for x in claims:
        qid=x["question_id"]; counts[qid]=counts.get(qid,0)
        if counts[qid] < 4:
            limited.append(x); counts[qid]+=1
    claims=limited
    write_jsonl(claims,claims_path)

    corpus=read_jsonl(args.corpus); texts=[x["title"]+" "+" ".join(x["abstract"]) for x in corpus]; ids=[int(x["doc_id"]) for x in corpus]
    scores=semantic_scores([x["claim"] for x in claims],texts,args.semantic_model,"cuda" if torch.cuda.is_available() else "cpu")
    retrieved=[]; model_claims=[]
    for i,c in enumerate(claims):
        ix=scores[i].argsort()[::-1][:args.top_k_docs]
        docs=[{"doc_id":ids[j],"title":corpus[j]["title"],"similarity":float(scores[i,j]),"abstract":corpus[j]["abstract"]} for j in ix]
        retrieved.append({"question_id":c["question_id"],"claim_id":c["claim_id"],"claim":c["claim"],"retrieved_documents":docs})
        model_claims.append({"id":i,"claim":c["claim"],"doc_ids":[ids[j] for j in ix],"retrieval":"semantic"})
    write_jsonl(retrieved,out/"retrieved_evidence.jsonl"); write_jsonl(model_claims,out/"claims_for_multivers.jsonl"); write_jsonl(corpus,out/"corpus.jsonl")
    pred=out/"multivers_raw_predictions.jsonl"; root=Path(__file__).resolve().parents[1]; repo=root/"multivers"
    cmd=[str(root/".venv/bin/python"),"multivers/predict.py","--checkpoint_path",str((root/args.checkpoint).resolve()),"--input_file",str((out/"claims_for_multivers.jsonl").resolve()),"--corpus_file",str((out/"corpus.jsonl").resolve()),"--output_file",str(pred.resolve()),"--batch_size","2","--device","0"]
    env=os.environ.copy(); env["PYTHONPATH"]=str(repo/"multivers")+os.pathsep+env.get("PYTHONPATH","")
    subprocess.run(cmd,cwd=repo,env=env,check=True)
    raw={int(x["id"]):x.get("evidence",{}) for x in read_jsonl(pred)}
    qmap={q["id"]:q for q in questions}; hits=[]
    results=[]; rows=[]
    for i,c in enumerate(claims):
        label,conf,doc=aggregate(raw.get(i,{})); evidence=[]
        for did,item in raw.get(i,{}).items():
            if item.get("sentences"): evidence.append({"doc_id":int(did),"label":item.get("label"),"sentences":item.get("sentences"),"confidence":max(item.get("rationale_probs",[]) or [0.0])})
        result={"question_id":c["question_id"],"claim_id":c["claim_id"],"claim":c["claim"],"predicted_label":label,"confidence":conf,"selected_document":doc,"selected_evidence":evidence}
        results.append(result); q=answers[c["question_id"]]; src=qmap[c["question_id"]]
        gold=set(src.get("gold_doc_ids",[])); top=set(model_claims[i]["doc_ids"])
        if gold: hits.append(bool(gold & top))
        rows.append([q["question"],q["generated_answer"],c["claim"],c.get("extraction_method",""),json.dumps(evidence,ensure_ascii=False),label,
                     src.get("source_claim",""),src.get("gold_label","")," ".join(map(str,sorted(gold))),"",""])
    write_jsonl(results,out/"verification_results.jsonl")
    with (out/"manual_review.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["question","MedGemma answer","claim","extraction method","retrieved evidence","MultiVerS prediction","source HealthVer claim","source claim gold label","source gold doc ids","human label","human evidence"]); w.writerows(rows)
    labels={x["predicted_label"]:0 for x in results}
    for x in results: labels[x["predicted_label"]]=labels.get(x["predicted_label"],0)+1
    report=f'''# Final MedGemma + MultiVerS evaluation\n\n## Dataset statistics\n\n- Questions: {len(questions)}\n- Generated answers: {len(answers)}\n- Extracted claims: {len(claims)}\n- Claims extracted by MedGemma JSON: {sum(1 for x in claims if x.get("extraction_method")=="medgemma_json")}\n- Claims from sentence fallback: {sum(1 for x in claims if x.get("extraction_method")=="safe_sentence_fallback")}\n- Claims whose top-{args.top_k_docs} retrieved documents include a gold document of the source HealthVer claim: {sum(hits)}/{len(hits)}\n- Semantic retrieval model: `{args.semantic_model}`\n- Verifier checkpoint: `{args.checkpoint}`\n\n## Automatic output distribution\n\n- SUPPORT: {labels.get("SUPPORT",0)}\n- CONTRADICT: {labels.get("CONTRADICT",0)}\n- NEI: {labels.get("NEI",0)}\n\n## Human-evaluated performance\n\nAccuracy, Macro-F1, evidence precision, evidence recall, evidence F1, and unsupported claim rate are **pending manual annotation**. The annotation file is `manual_review.csv`; human labels must be filled before calculating these metrics.\n\n| System | Accuracy | Macro-F1 |\n|---|---:|---:|\n| MedGemma only | Pending human review | Pending |\n| MedGemma + original MultiVerS | Not run by design | Not run |\n| MedGemma + fine-tuned MultiVerS | Pending human review | Pending |\n\n## Error analysis form\n\nFor each incorrect claim, mark the primary error as: MedGemma hallucination, claim extraction error, retrieval error, or MultiVerS verification error.\n'''
    (out/"final_report.md").write_text(report,encoding="utf-8")
    print(json.dumps({"questions":len(questions),"answers":len(answers),"claims":len(claims),"output_dir":str(out)},indent=2))

if __name__ == "__main__": main()
