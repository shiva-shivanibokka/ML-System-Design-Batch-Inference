# TODO — remaining work to take this live

The code is migrated to the real **KKBox** dataset, Gradio is replaced by a
**Next.js** dashboard, and the deploy target is **Vercel (API + dashboard) +
Neon (Postgres) + GitHub Actions (nightly cron + CI)**. What's left is the work
that needs *your* accounts, GPU, and the real data — plus a few polish items.

Legend: ⬜ not started · 🧪 needs your testing (I couldn't run it here)

---

## A. Get it running locally (needs Kaggle + your GPU)

- ✅ **Kaggle rules accepted** (via browser) + token works.
- ✅ **Download data**: `python data/download_kkbox.py` — fetches + extracts the `.7z` files (needs `py7zr`).
- ✅ **Build the full dataset**: `python data/build_dataset.py` → `data/customers.parquet` (**970,960 rows, 9% churn, 48 MB**). ETL verified on the real files.
- ✅ **Train**: `python models/train.py --eval` → **AUC-ROC 0.812**, artifacts in `models/`.
- ✅ **End-to-end scoring proven**: `python score_batch.py --skip-postgres` scored all 970,960 rows in ~10s, validation gates passed.
- ⬜ **Build the committed sample** (small real slice the nightly CI scores):
  `python data/build_dataset.py --limit 20000 --output data/sample/customers.parquet` then `git add data/sample/customers.parquet`
- ⬜ **Commit model artifacts** for nightly CI (git-ignored by default):
  `git add -f models/churn_model.pkl models/label_encoders.pkl models/feature_columns.pkl`
- 🧪 **Local full stack**: copy `.env.example` → `.env`, then `docker-compose up --build`. Verify the Airflow DAG + Spark broadcast path on the real parquet. *(Not yet run — the pandas path above is proven; Spark/Airflow still to test.)*
- 🧪 **Local API + dashboard**: `uvicorn serving.app:app --port 8000`, then in `dashboard/`: `npm install`, `npm run dev` (commit the generated `package-lock.json`).

## B. Deploy (all free tiers)

- ⬜ **Neon**: create a project → copy the **pooled** connection string → apply schema: `psql "<DATABASE_URL>" -f db/schema.sql`.
- ⬜ **Vercel — API project**: import this repo, Root Directory = `.` (repo root, uses `vercel.json` + `api/index.py`). Env: `DATABASE_URL`, `CORS_ORIGINS`. Note the deployed URL.
- ⬜ **Vercel — dashboard project**: same repo, Root Directory = `dashboard`. Env: `NEXT_PUBLIC_API_URL` = the API URL above.
- ⬜ **GitHub secret**: add `DATABASE_URL` (repo → Settings → Secrets) so `nightly.yml` can write to Neon.
- 🧪 **First nightly run**: trigger `Nightly batch scoring` manually (Actions tab → Run workflow) once the model + `data/sample/customers.parquet` are committed.

## C. Verify (things I built but could not run here)

- 🧪 **Neon + asyncpg**: confirm the API connects (the `VERCEL` branch in `db/connection.py` sets `statement_cache_size=0` + `ssl=require` for the pooled endpoint) — this is the most likely first-deploy snag.
- 🧪 **Dashboard charts** render against the live API (Recharts log-scale axis on the Benchmark tab especially).
- 🧪 **CI**: push a branch and confirm both jobs (Python lint+test, Next.js build) pass.

## D. Deferred polish / known gaps (optional, nice-to-have)

- ⬜ **Design pass on the dashboard** — the UI is clean but functional; a `dataviz` / `frontend-design` pass would sharpen the palette, spacing, and empty states.
- ⬜ **ADR docs** — the README covers key design decisions; a short `docs/adr/` (why pre-compute, why KKBox, why the right-sized engine) would signal system-design explicitly.
- ⬜ **Model registry / retrain trigger** — currently the model is a committed artifact; a lightweight version bump on drift (PSI > 0.2) would close the MLOps loop.
- ⬜ **Airflow parity with the deployed path** — the DAG still uses Spark + its own validation task; the deployed path is `score_batch.py` (pandas). They share `features.py` and the validation gates are equivalent, but the two entry points could be unified further.

---

*Status is mirrored in the README's Roadmap section. Update both when you close an item.*
