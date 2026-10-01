"""Central configuration. Edit values here, nowhere else."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_CSV = ROOT / "data" / "raw" / "LLM_Calibration_1100_Questions_labeled.csv"
PROCESSED_CSV = ROOT / "data" / "processed" / "questions.csv"
OUT_DIR = ROOT / "outputs"

# ---- Models (check OpenAI's model list; names change over time) ----------
ANSWER_MODEL = "gpt-4o"      # must be a NON-reasoning chat model that returns logprobs
GRADER_MODEL = "gpt-4.1"     # judge; keep it fixed for every answer model

# ---- Inference settings (record these in the paper, Section 3.10) --------
TEMPERATURE = 0
SEED = 42
MAX_TOKENS = 64
WORKERS = 8                  # parallel requests; lower this if you hit rate limits

# ---- Analysis settings ---------------------------------------------------
N_BINS = 10
MIN_GROUP_N = 30             # groups smaller than this are merged / get no ECE
N_BOOTSTRAP = 1000

SYSTEM_PROMPT_LOGPROB = (
    "You are a factual question-answering system. Reply with only the answer "
    "itself: a short phrase, name, number or date. No explanation and no full "
    "sentence. Always give your single best guess, even if you are unsure."
)

SYSTEM_PROMPT_VERBAL = (
    "You are a factual question-answering system. Reply ONLY with JSON of the "
    'form {"answer": "<short answer>", "confidence": <integer 0-100>}, where '
    "confidence is your estimated probability (in percent) that the answer is "
    "correct. Always give your single best guess, even if you are unsure."
)

GRADER_PROMPT = """You are grading a predicted answer to a factual question against a gold target.
Assign exactly one grade:
A = CORRECT: the prediction fully contains the important information in the gold target and contradicts nothing in it. Hedging is fine if the gold answer is clearly the answer given. Minor spelling variants of names are fine. Numbers must be correct to the precision given in the gold target.
B = INCORRECT: the prediction contradicts the gold target, or gives a different answer, even if hedged.
C = NOT_ATTEMPTED: the prediction does not give the gold information and does not contradict it (for example "I don't know").

Question: {question}
Gold target: {gold}
Predicted answer: {pred}

Reply with a single letter: A, B or C."""


def run_dir(model: str) -> Path:
    d = OUT_DIR / model.replace("/", "_")
    d.mkdir(parents=True, exist_ok=True)
    return d


def paths(model: str, mode: str) -> dict:
    d = run_dir(model)
    return {
        "dir": d,
        "answers": d / f"answers_{mode}.jsonl",
        "graded": d / f"graded_{mode}.jsonl",
        "audit": d / f"audit_sample_{mode}.csv",
        "meta": d / f"run_meta_{mode}.json",
        "analysis": d / f"analysis_{mode}",
    }
