# HDFC Mutual Fund Facts Assistant

A lightweight facts-only RAG chatbot that answers short, factual questions about five
HDFC mutual funds using **only** their public Groww pages. Grounded generation (Mistral)
over a persisted ChromaDB vector index, with strict safety gates (no advice, no PII, no
unsupported performance claims, no out-of-scope answers).

## Scope

The five approved funds (all Direct Growth):

| Fund | Category | Source |
|---|---|---|
| HDFC Large Cap | large-cap | `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth` |
| HDFC Flexi Cap (legacy "Equity" slug) | flexi-cap | `https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth` |
| HDFC ELSS Tax Saver | elss | `https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth` |
| HDFC Small Cap | small-cap | `https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth` |
| HDFC Balanced Advantage | hybrid | `https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth` |

## Repository layout

```text
code/
  app.py                 Streamlit UI + orchestration
  config.py              Allowlist, paths, thresholds
  ingestion/             Load -> clean -> chunk -> embed/upsert (ChromaDB)
  rag/                   Policy gate, resolver, retriever, prompt, generator, validator
  evaluate.py            Phase 6/7 evaluation harness (--calibrate, --live)
data/                    Corpus + persisted ChromaDB index + threshold
  chroma/                Committed vector index (for ephemeral-FS hosts)
  threshold.json         Relevance threshold (0.55)
.hfcache/                Committed offline MiniLM model cache
docs/                    PRD, architecture
tests/                   Pytest suite + labelled retrieval dataset
```

## Setup (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in MISTRAL_API_KEY
```

## Build the index (one-time, per environment)

```bash
python -m code.ingestion            # Phase 1: fetch the five approved pages
python -m code.ingestion --chunk    # Phase 2: chunk the cleaned documents
python -m code.ingestion --index    # Phase 4: embed + upsert into ChromaDB
```

Phase 3 (embedding, `python -m code.ingestion --embed`) is optional: `--index` already
embeds the chunks as part of upsert. The standalone `--embed` step only *exports* the
vectors to `data/embeddings/` if you want that artifact; it is not required to run the
app.

The ChromaDB index and MiniLM model cache are committed so hosts with an ephemeral
filesystem (Streamlit Community Cloud, Render free) can run without a build step.

## Run the app

```bash
python -m streamlit run code/app.py --server.port 8501
```

Open http://localhost:8501. Example in the UI:

- What is the expense ratio of HDFC Large Cap Fund Direct Growth?
- What is the lock-in period for the HDFC ELSS fund?
- What is the minimum SIP for HDFC Small Cap Fund Direct Growth?

## Test

```bash
python -m pytest -q
```

## Evaluation

```bash
python -m code.evaluate                    # deterministic retrieval/safety/recall@4
python -m code.evaluate --live             # end-to-end unsupported-claim check (needs MISTRAL_API_KEY)
python -m code.evaluate --calibrate        # pick a relevance threshold -> data/threshold.json
```

## Deploy: Streamlit Community Cloud (current)

This app is deployed on **Streamlit Community Cloud**.

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io -> **Create app** -> the `GrowChatBot` repo,
   branch `main`, main file `code/app.py`.
3. In **Advanced settings -> Secrets**, add:

   ```toml
   MISTRAL_API_KEY = "<your-key>"
   ```

   (Real `secrets.toml` is gitignored; see `.streamlit/secrets.toml.example`.)
4. Deploy and open your live URL.

### Re-deploying after a change

Streamlit Cloud does **not** auto-pull every push by default. After pushing new code,
open the app on **share.streamlit.io**, use the app card's overflow menu
(`⋮` -> **Rerun** / **Reboot**), or switch the deployed branch away and back. To make
it automatic, enable auto-deploy in the app's **Settings**.

> Free-tier caveat: apps sleep after ~72h idle and cold-start on the next visit, so
> the first load after sleep is slow (it loads the ~86MB MiniLM model from the
> committed cache — it will not re-download). Subsequent questions are fast.

## Deploy: Render (alternative)

`render.yaml` + `build.sh`/`start.sh` are still included as an alternative host. In
the Render dashboard use **Manual Deploy -> Deploy latest commit** and set
`MISTRAL_API_KEY` as a secret (`sync: false`). Free instances also sleep on idle with
the same cold-start caveat.

## Notes

- `MISTRAL_API_KEY` is the only secret. It is read from the environment, `.env`, or
  Streamlit secrets — never committed.
- Retrieval uses a calibrated relevance threshold (0.55, `data/threshold.json`) and
  top-4. Unsafe responses (advice, performance, PII, out-of-scope) are blocked before
  any embedding or LLM call.
- Model cache lives in `.hfcache/`, ChromaDB in `data/chroma/`, both committed for
  ephemeral-FS hosting. Re-running index builds locally will dirty those committed
  files (SQLite/noise); restore with `git restore --staged --worktree data/chroma .hfcache`
  before committing if the index itself didn't change.
