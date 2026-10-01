"""Step 2: ask the model every question and record answer + confidence.

Modes
  logprob : confidence from token probabilities (primary method in the paper)
  verbal  : model states its own confidence 0-100 (optional comparison)

Resumable: already-answered ids are skipped, so you can stop and re-run safely.
Pilot first:  python 02_run_inference.py --limit 20
"""
import argparse, json, platform, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import openai
from openai import OpenAI, BadRequestError
from tqdm import tqdm
import config


def token_stats(content):
    """Confidence variants from the logprob content of the answer."""
    toks = [t for t in content if t.token.strip() not in ("", ".")] or list(content)
    lps = np.array([t.logprob for t in toks])
    probs = np.exp(lps)
    return {
        "conf_mean": float(probs.mean()),          # primary (paper Section 3.4)
        "conf_geo": float(np.exp(lps.mean())),     # geometric mean
        "conf_first": float(probs[0]),             # first answer token
        "conf_min": float(probs.min()),            # weakest token
        "n_tokens": len(toks),
    }


def ask(client, args, row):
    system = config.SYSTEM_PROMPT_VERBAL if args.mode == "verbal" else config.SYSTEM_PROMPT_LOGPROB
    kwargs = dict(
        model=args.model,
        temperature=config.TEMPERATURE,
        seed=config.SEED,
        max_tokens=config.MAX_TOKENS,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": row["question"]}],
    )
    if args.mode == "logprob":
        kwargs["logprobs"] = True
    else:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(6):
        try:
            r = client.chat.completions.create(**kwargs)
            break
        except BadRequestError:
            raise                       # wrong model/parameter: retrying will not help
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)

    ch = r.choices[0]
    raw = (ch.message.content or "").strip()
    rec = {"id": row["id"], "model": r.model, "finish_reason": ch.finish_reason,
           "system_fingerprint": r.system_fingerprint,
           "timestamp": datetime.now(timezone.utc).isoformat()}

    if args.mode == "logprob":
        if not (ch.logprobs and ch.logprobs.content):
            raise RuntimeError("API returned no logprobs for this model.")
        rec["model_answer"] = raw
        rec.update(token_stats(ch.logprobs.content))
    else:
        try:
            data = json.loads(raw)
            rec["model_answer"] = str(data.get("answer", "")).strip()
            rec["conf_verbal"] = float(np.clip(float(data["confidence"]) / 100, 0, 1))
        except Exception:
            rec["model_answer"], rec["conf_verbal"] = raw, None
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.ANSWER_MODEL)
    ap.add_argument("--mode", choices=["logprob", "verbal"], default="logprob")
    ap.add_argument("--limit", type=int, default=None, help="only process N new questions")
    args = ap.parse_args()

    p = config.paths(args.model, args.mode)
    qs = pd.read_csv(config.PROCESSED_CSV)
    done = set()
    if p["answers"].exists():
        with open(p["answers"]) as fh:
            done = {json.loads(l)["id"] for l in fh if l.strip()}
    todo = [r for r in qs.to_dict("records") if r["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} already done, {len(todo)} to run with {args.model} [{args.mode}]")
    if not todo:
        return

    meta = json.loads(p["meta"].read_text()) if p["meta"].exists() else {"runs": []}
    meta["runs"].append({
        "started_utc": datetime.now(timezone.utc).isoformat(), "model_requested": args.model,
        "mode": args.mode, "temperature": config.TEMPERATURE, "seed": config.SEED,
        "max_tokens": config.MAX_TOKENS,
        "system_prompt": config.SYSTEM_PROMPT_VERBAL if args.mode == "verbal" else config.SYSTEM_PROMPT_LOGPROB,
        "openai_sdk": openai.__version__, "python": platform.python_version()})
    p["meta"].write_text(json.dumps(meta, indent=2))

    client = OpenAI()                   # reads OPENAI_API_KEY from the environment
    failed = []
    with ThreadPoolExecutor(config.WORKERS) as ex, open(p["answers"], "a") as out:
        futs = {ex.submit(ask, client, args, r): r["id"] for r in todo}
        for f in tqdm(as_completed(futs), total=len(futs)):
            try:
                rec = f.result()
            except BadRequestError as e:
                print(f"\nAPI rejected the request: {e}\nThis model probably does not support "
                      "logprobs/temperature. Use a non-reasoning chat model.")
                for x in futs:
                    x.cancel()
                sys.exit(1)
            except Exception as e:
                failed.append((futs[f], str(e)))
                continue
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
    if failed:
        print(f"{len(failed)} failed (re-run the script to retry), e.g. {failed[:3]}")


if __name__ == "__main__":
    main()
