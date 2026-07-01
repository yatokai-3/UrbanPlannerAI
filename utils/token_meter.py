"""
Lightweight, process-wide Groq token meter.

Every agent's `chat_json` (and Agent 1's extraction calls) pass their Groq
response here via `record()`. It accumulates prompt/completion/total tokens
overall AND per-model (important because Groq's daily limit is PER MODEL).

Usage:
    from utils import token_meter
    token_meter.reset()          # at the start of a run
    ...                          # agents call token_meter.record(response)
    token_meter.report()         # at the end
"""

_overall = {"calls": 0, "prompt": 0, "completion": 0, "total": 0}
_by_model = {}   # model name -> same dict shape


def reset():
    """Zero the counters (call at the start of a pipeline run)."""
    global _overall, _by_model
    _overall = {"calls": 0, "prompt": 0, "completion": 0, "total": 0}
    _by_model = {}


def record(response):
    """Add one Groq response's token usage to the running totals. Safe if usage missing."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    model = getattr(response, "model", "unknown")
    p = getattr(usage, "prompt_tokens", 0) or 0
    c = getattr(usage, "completion_tokens", 0) or 0
    t = getattr(usage, "total_tokens", 0) or (p + c)

    bucket = _by_model.setdefault(model, {"calls": 0, "prompt": 0, "completion": 0, "total": 0})
    for store in (_overall, bucket):
        store["calls"] += 1
        store["prompt"] += p
        store["completion"] += c
        store["total"] += t


def snapshot() -> dict:
    """Return a copy of the current totals (overall + per-model)."""
    return {"overall": dict(_overall), "by_model": {m: dict(v) for m, v in _by_model.items()}}


def report(label: str = "TOKEN USAGE"):
    """Print a human-readable summary, including a rough free-tier (100k TPD) gauge."""
    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)
    print(f"  Total LLM calls : {_overall['calls']}")
    print(f"  Prompt tokens   : {_overall['prompt']:,}")
    print(f"  Completion tokens: {_overall['completion']:,}")
    print(f"  TOTAL tokens    : {_overall['total']:,}")
    if _by_model:
        print("  --- per model (each has its own ~100k/day free-tier pool) ---")
        for model, v in _by_model.items():
            pct = v["total"] / 100_000 * 100
            print(f"    {model:32} {v['total']:>8,} tokens  ({v['calls']} calls, ~{pct:.0f}% of a 100k/day pool)")
    print("=" * 60)
