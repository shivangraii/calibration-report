"""Step 3: grade each answer CORRECT / INCORRECT / NOT_ATTEMPTED.

Exact normalised matches are graded locally; everything else goes to the grader
model. Also creates a 100-row audit sheet for manual verification.
"""
import argparse, json, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import config

LABELS = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}


def norm(s):
    s = re.sub(r"[^\w\s]", "", str(s).lower())
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return " ".join(s.split())


def grade_one(client, grader, row):
    pred = str(row["model_answer"]).strip()
    base = {"id": row["id"], "grader_model": grader}
    if not pred:
        return {**base, "grade": "NOT_ATTEMPTED", "grade_method": "empty", "grader_raw": ""}
    if norm(pred) == norm(row["reference"]):
        return {**base, "grade": "CORRECT", "grade_method": "exact", "grader_raw": ""}
    prompt = config.GRADER_PROMPT.format(question=row["question"], gold=row["reference"], pred=pred)
    for _ in range(4):
        r = client.chat.completions.create(
            model=grader, temperature=0, max_tokens=5,
            messages=[{"role": "user", "content": prompt}])
        raw = (r.choices[0].message.content or "").strip()
        if raw[:1].upper() in LABELS:
            return {**base, "grade": LABELS[raw[:1].upper()], "grade_method": "llm", "grader_raw": raw}
    return {**base, "grade": "UNPARSED", "grade_method": "llm", "grader_raw": raw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.ANSWER_MODEL)
    ap.add_argument("--mode", choices=["logprob", "verbal"], default="logprob")
    ap.add_argument("--grader", default=config.GRADER_MODEL)
    args = ap.parse_args()

    p = config.paths(args.model, args.mode)
    qs = pd.read_csv(config.PROCESSED_CSV)[["id", "question", "reference"]]
    ans = pd.read_json(p["answers"], lines=True)
    df = ans.merge(qs, on="id")

    done = set()
    if p["graded"].exists():
        with open(p["graded"]) as fh:
            done = {json.loads(l)["id"] for l in fh if l.strip()}
    todo = [r for r in df.to_dict("records") if r["id"] not in done]
    print(f"{len(done)} already graded, {len(todo)} to grade with {args.grader}")

    if todo:
        client = OpenAI()
        with ThreadPoolExecutor(config.WORKERS) as ex, open(p["graded"], "a") as out:
            futs = [ex.submit(grade_one, client, args.grader, r) for r in todo]
            for f in tqdm(as_completed(futs), total=len(futs)):
                try:
                    out.write(json.dumps(f.result(), ensure_ascii=False) + "\n")
                    out.flush()
                except Exception as e:
                    print("grading error:", e, file=sys.stderr)

    g = pd.read_json(p["graded"], lines=True)
    print(g["grade"].value_counts().to_string())

    if not p["audit"].exists():
        s = df.merge(g[["id", "grade"]], on="id").sample(n=min(100, len(df)), random_state=config.SEED)
        s = s.rename(columns={"grade": "auto_grade"})
        s["human_grade"] = ""
        s[["id", "question", "reference", "model_answer", "auto_grade", "human_grade"]].to_csv(p["audit"], index=False)
        print(f"Audit sheet -> {p['audit']}  (fill human_grade with CORRECT / INCORRECT / NOT_ATTEMPTED)")


if __name__ == "__main__":
    main()
