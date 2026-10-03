#!/usr/bin/env python3
"""Calculate final metrics after manual_review.csv is annotated.

Fill `human label` with SUPPORT, CONTRADICT, or NEI. Put human evidence in
JSON such as [{"doc_id": 57, "sentences": [1, 2]}].
"""
import argparse, csv, json
from collections import Counter
from pathlib import Path

LABELS=["SUPPORT","CONTRADICT","NEI"]
def f1(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
    return 2*p*r/(p+r) if p+r else 0

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--csv",default="results/medgemma_eval/manual_review.csv"); ap.add_argument("--output",default="results/medgemma_eval/human_metrics.json"); a=ap.parse_args()
    rows=list(csv.DictReader(Path(a.csv).open(encoding="utf-8")))
    if any(x["human label"].strip() not in LABELS for x in rows): raise SystemExit("Fill every human label first.")
    cm={g:{p:0 for p in LABELS} for g in LABELS}; correct=0; ec=er=eg=0; unsupported=0
    for x in rows:
        g=x["human label"].strip(); p=x["MultiVerS prediction"].strip(); cm[g][p]+=1; correct+=g==p
        gold={(int(e["doc_id"]),int(s)) for e in json.loads(x["human evidence"] or "[]") for s in e.get("sentences",[])}
        pred={(int(e["doc_id"]),int(s)) for e in json.loads(x["retrieved evidence"] or "[]") for s in e.get("sentences",[])}
        ec+=len(gold&pred); er+=len(pred); eg+=len(gold); unsupported+=g=="NEI" and bool(pred)
    per={};
    for l in LABELS:
        tp=cm[l][l]; fp=sum(cm[g][l] for g in LABELS if g!=l); fn=sum(cm[l][p] for p in LABELS if p!=l); per[l]=f1(tp,fp,fn)
    ep=ec/er if er else 0; ee=ec/eg if eg else 0
    out={"n_claims":len(rows),"accuracy":correct/len(rows),"macro_f1":sum(per.values())/3,"evidence_precision":ep,"evidence_recall":ee,"evidence_f1":2*ep*ee/(ep+ee) if ep+ee else 0,"unsupported_claim_rate":unsupported/len(rows),"per_label_f1":per,"confusion":cm}
    Path(a.output).write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2))
if __name__=="__main__": main()
