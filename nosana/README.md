# Nosana self-deploy (vLLM)

Example job definition for serving **Qwen2.5-7B-Instruct** with an OpenAI-compatible
API via vLLM on Nosana.

## Uncertainty / TODO-verify

- Confirmed against Nosana docs (`learn.nosana.com`) as of article prep:
  - Schema version `"0.1"`, `type: "container"`, ops of `container/run`
  - `gpu: true`, `expose` as number or object with `health_checks`
  - Image pattern `docker.io/vllm/vllm-openai:...`
- **Uncertain / verify before posting a job:**
  - Exact market ID for your GPU tier (`nosana market list`)
  - Whether HF model download on first boot is allowed without an S3 `resources`
    cache entry (official Llama examples often mount `models.nosana.io` caches)
  - Image tag (`v0.6.4` here is an example; pin to a tag Nosana nodes pull reliably)
  - Public URL shape returned after `nosana job post` — set that as `NOSANA_SELF_URL`
    (include `/v1` if the gateway does not already)

## Post (CLI)

```bash
# pick a market with enough VRAM for 7B
nosana market list
nosana job post --file nosana/job.json --market <MARKET_ID>
```

Then point the harness at the exposed endpoint:

```bash
export NOSANA_SELF_URL="https://<node-or-gateway-host>/v1"
# optional auth if your deployment requires it
export NOSANA_API_KEY="..."
python run.py  # or with other keys
```

Alternatively use Nosana's hosted OpenAI-compatible inference API
(`https://inference.nosana.com/v1`) with `NOSANA_API_KEY` and the `nosana` provider
in `config.yaml` — no job JSON required.
