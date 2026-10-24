# Refusal Bench

Benchmark harness for a HackerNoon **Decentralize AI** hackathon article comparing
**refusals, cost, and latency** across five closed AI APIs (OpenAI, Anthropic, Google
Gemini, xAI, Mistral) and open-weight models on **Nosana** (hosted inference + optional
self-deployed vLLM job).

> **Do not call paid APIs until you have keys and budget.** The default workflow is
> `--dry-run`, which mocks responses and still produces judgments + charts end-to-end.

## Layout

| Path | Purpose |
|------|---------|
| `prompts.csv` | 142 prompts across 8 sensitive-but-legitimate categories + should_refuse + lookalikes |
| `config.yaml` | Model IDs, shared system prompt, temperature 0, max 512 tokens |
| `prices.yaml` | Placeholder USD / 1M token prices (`TODO-verify`) |
| `rubric.md` | refuse / partial / comply rubric for the judge |
| `run.py` | Async harness → `results.jsonl` (TTFT, latency, tokens, errors) |
| `judge.py` | Label responses; export 20% `handcheck_sample.csv` |
| `analyze.py` | Metrics + PNG charts under `charts/` |
| `nosana/job.json` | Example Nosana vLLM job (Qwen2.5-7B-Instruct) |

## Setup

```bash
cd refusal-bench
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment variables

| Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GEMINI_API_KEY` | Google Gemini (OpenAI-compatible endpoint) |
| `XAI_API_KEY` | xAI |
| `MISTRAL_API_KEY` | Mistral |
| `NOSANA_API_KEY` | Nosana hosted inference (`https://inference.nosana.com/v1`) |
| `NOSANA_SELF_URL` | Optional base URL of a self-deployed vLLM job (…`/v1`) |

Providers **without** a key are skipped (except in `--dry-run`, which mocks all configured providers).

## Dry-run (no keys, no paid calls)

```bash
source .venv/bin/activate
python run.py --dry-run
python judge.py --dry-run
python analyze.py
```

Outputs:

- `results.jsonl` — raw per-call records
- `judgments.jsonl` — labels
- `handcheck_sample.csv` — ~20% stratified sample for manual review
- `summary.json` — metrics
- `charts/*.png` — stacked bar, heatmap, scatter, latency box, cost bars

## Live run (when keys exist)

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
# etc.
python run.py
python judge.py          # uses judge model from config if OPENAI_API_KEY set
python analyze.py
```

Optional: `--limit N` on `run.py` for a smoke test; `--concurrency 8` to tune parallelism.

## Nosana

1. **Hosted:** set `NOSANA_API_KEY`; provider `nosana` in `config.yaml` targets
   `https://inference.nosana.com/v1`.
2. **Self-deploy:** see `nosana/job.json` and `nosana/README.md`. After the job is up,
   set `NOSANA_SELF_URL` to the OpenAI-compatible base URL.

## Metrics definitions

- **Refusal rate** — share of responses labeled `refuse`
- **Over-refusal** — `refuse` on `should_answer` or `benign_lookalike`
- **Compliance on should_refuse** — `comply` when the prompt should have been refused (safety failure)
- **Latency** — TTFT (streaming) and total wall time
- **Cost** — from token counts × `prices.yaml` (placeholders)

## Article note

All `should_refuse` prompts are clearly harmful in intent but **not** operationally detailed.
Benign lookalikes (e.g. “kill a Python process”) stress-test over-refusal.
