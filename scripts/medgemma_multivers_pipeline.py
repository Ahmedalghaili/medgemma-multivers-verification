#!/usr/bin/env python3
"""MedGemma answer -> claims -> frozen MultiVerS verification pipeline.

MedGemma is gated on Hugging Face. After access is granted, authenticate with
`huggingface-cli login` and run this script with the model ID. MultiVerS does
not retrieve documents; the supplied corpus is the evidence candidate set.
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from transformers import AutoModel, AutoTokenizer


def generate(processor, model, prompt, max_new_tokens=512):
    if hasattr(processor, "apply_chat_template"):
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        inputs = processor(text=prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = output[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def parse_json(text):
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("MedGemma did not return a JSON object:\n" + text)
    return json.loads(match.group(0))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).open(encoding="utf-8")]


def dump_jsonl(rows, path):
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def semantic_scores(queries, documents, model_name, device, batch_size=16):
    """Cosine similarities using a biomedical Transformer and mean pooling."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device).eval()

    def encode(texts):
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = tokenizer(texts[start:start + batch_size], padding=True,
                                  truncation=True, max_length=512,
                                  return_tensors="pt").to(device)
                pooled = mean_pool(encoder(**batch).last_hidden_state,
                                   batch["attention_mask"])
                chunks.append(torch.nn.functional.normalize(pooled, dim=1).cpu())
        return torch.cat(chunks)

    query_vectors = encode(queries)
    document_vectors = encode(documents)
    scores = query_vectors @ document_vectors.T
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    ap.add_argument("--corpus_file", required=True)
    ap.add_argument("--output_dir", default="results/medgemma_pipeline")
    ap.add_argument("--model_id", default="google/medgemma-4b-it")
    ap.add_argument("--checkpoint", default="multivers/finetune_results/healthver_finetuned_resume/checkpoint/epoch=3-step=2648.ckpt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--top_k_docs", type=int, default=10)
    ap.add_argument("--retrieval", choices=["tfidf", "semantic"], default="tfidf")
    ap.add_argument("--semantic_model", default="pritamdeka/S-PubMedBert-MS-MARCO")
    ap.add_argument("--semantic_batch_size", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    corpus = load_jsonl(args.corpus_file)
    corpus_texts = [row["title"] + " " + " ".join(row["abstract"]) for row in corpus]
    corpus_ids = [int(row["doc_id"]) for row in corpus]
    vectorizer = TfidfVectorizer(stop_words="english")
    corpus_matrix = vectorizer.fit_transform(corpus_texts)

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    answer_prompt = f"""You are a medical assistant. Answer the user's question carefully.
Use cautious language and do not invent citations.

User question:
{args.question}

Return only the answer text."""
    answer = generate(processor, model, answer_prompt, args.max_new_tokens)

    claim_prompt = f"""Extract atomic, medically meaningful factual claims from the answer below.
Preserve every uncertainty or qualification in the original answer, including words such as
may, might, can, could, associated with, suggests, appears to, and evidence is limited.
Never make a claim stronger than the answer. Split compound statements into atomic claims.
Do not extract advice, greetings, questions, or unsupported additions.
Return only valid JSON in this exact format:
{{"claims": [{{"claim": "one atomic claim"}}]}}

Answer:
{answer}"""
    extracted = parse_json(generate(processor, model, claim_prompt, args.max_new_tokens))
    claims = [str(x.get("claim", "")).strip() for x in extracted.get("claims", [])
              if str(x.get("claim", "")).strip()]
    if not claims:
        raise ValueError("No claims were extracted from the MedGemma answer.")

    claims_file = out / "claims.jsonl"
    claim_rows = []
    semantic_matrix = None
    if args.retrieval == "semantic":
        retrieval_device = "cuda" if torch.cuda.is_available() else "cpu"
        semantic_matrix = semantic_scores(claims, corpus_texts, args.semantic_model,
                                          retrieval_device, args.semantic_batch_size)
    for i, claim in enumerate(claims):
        if semantic_matrix is None:
            scores = linear_kernel(vectorizer.transform([claim]), corpus_matrix).ravel()
        else:
            scores = semantic_matrix[i]
        top_indices = scores.argsort()[::-1][: args.top_k_docs]
        claim_rows.append({
            "id": i,
            "claim": claim,
            "doc_ids": [corpus_ids[j] for j in top_indices],
            "retrieval": args.retrieval,
            "retrieval_scores": [float(scores[j]) for j in top_indices],
        })
    dump_jsonl(claim_rows, claims_file)
    corpus_file = out / "corpus.jsonl"
    dump_jsonl(corpus, corpus_file)
    raw_file = out / "medgemma_answer.json"
    raw_file.write_text(json.dumps({"question": args.question, "answer": answer,
                                    "claims": claims, "retrieval": args.retrieval},
                                   indent=2) + "\n", encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[1] / "multivers"
    prediction_file = out / "verification_predictions.jsonl"
    command = [
        str(Path(__file__).resolve().parents[1] / ".venv/bin/python"),
        "multivers/predict.py",
        "--checkpoint_path", str(Path(args.checkpoint).resolve()),
        "--input_file", str(claims_file.resolve()),
        "--corpus_file", str(corpus_file.resolve()),
        "--output_file", str(prediction_file.resolve()),
        "--batch_size", "2",
        "--device", "0",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "multivers") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, cwd=repo_root, env=env, check=True)

    predictions = {int(row["id"]): row.get("evidence", {}) for row in load_jsonl(prediction_file)}
    report = ["# Evidence-grounded medical answer", "", "## Answer", "", answer, "", "## Claim verification", ""]
    for i, claim in enumerate(claims):
        evidence = predictions.get(i, {})
        report.extend([f"### Claim {i + 1}", "", claim, ""])
        evidence_items = [
            (doc_id, item) for doc_id, item in evidence.items()
            if item.get("sentences")
        ][:3]
        if not evidence_items:
            report.extend(["**Verdict:** NEI", ""])
            continue
        for doc_id, item in evidence_items:
            report.extend([f"**Verdict:** {item['label']}", f"**Document:** {doc_id}", f"**Evidence sentences:** {item['sentences']}", ""])
    (out / "evidence_grounded_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"answer_file": str(raw_file), "claims_file": str(claims_file), "predictions_file": str(prediction_file), "report_file": str(out / 'evidence_grounded_report.md')}, indent=2))


if __name__ == "__main__":
    main()
