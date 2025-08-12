# README — CSC 4792 Group Project (Team N)

**Topic 2.1.10 — Classify each MP’s response as *relevant* or *not relevant* to the motion**

> All deliverables must follow the CRISP-DM methodology and the course’s naming/commit conventions.

---

## 0) TL;DR for the team

* **Goal:** Given a motion and the debate transcript for a parliamentary sitting, label every MP utterance as **Relevant** or **NotRelevant** to that motion.
* **Checkpoints (in order):** **\[BU] → \[DU] → \[DP] → \[MO] → \[EV] → \[DE]**
* **Final submissions:** technical report PDF, GitHub repo, Colab notebook, slide deck. Use the exact filenames below.

---

## 1) Problem statement

Classify each **speaker turn** in Debates/Proceedings as **Relevant** (argues for/against the motion, provides evidence, proposes actions) or **NotRelevant** (procedural points, greetings, tangents, jokes).
**Output:** CSV with `(sitting_id, speaker, timestamp, text, label, confidence)` plus a confusion matrix and slice metrics (per sitting & per speaker).

Why it matters: helps Hansard indexing, search, and civic analysis.

---

## 2) Deliverables & naming (strict)

* **Report (PDF, ACM template, ≤4 pages):**
  `docs-unza25-csc4792-project_team_16-technical_report.pdf`
* **GitHub repository:**
  `code-unza25-csc4792-project_team_16-repository`
* **Google Colab notebook:**
  `code-unza25-csc4792-project_team_16-notebook.ipynb`
* **Slides (PDF):**
  `talks-unza25-csc4792-project_team_16-presentation_slides.pdf`
* **Everything follows CRISP-DM**; checkpoints are graded and tracked via Git/Colab activity.

**Marking snapshot:** Project management (Git/Colab activity), dataset quality, implementation/demonstration, presentation, and technical report quality.

---

## 3) Repo structure

```
code-unza25-csc4792-project_team_16-repository/
├── README.md
├── data/
│   ├── raw/              # scraped HTML/PDF
│   ├── interim/          # parsed JSONL utterances
│   └── processed/        # train/val/test splits, features
├── docs/
│   └── report_assets/    # figs/tables used in paper
├── notebooks/
│   └── code-unza25-csc4792-project_team_16-notebook.ipynb
├── src/
│   ├── scrape/           # parliament scraping
│   ├── parse/            # segmentation & linkage to motion
│   ├── label/            # annotation utilities & QA
│   ├── features/         # vectorizers/tokenizers
│   ├── models/           # baselines & transformers
│   ├── eval/             # metrics, error analysis
│   └── app/              # CLI + simple demo
├── experiments/
│   └── runs/             # metrics.json, confusion matrices
├── reports/
│   ├── docs-unza25-csc4792-project_team_16-technical_report.pdf
│   └── talks-unza25-csc4792-project_team_16-presentation_slides.pdf
└── pyproject.toml | requirements.txt | Makefile
```

---

## 4) Setup

* **Python:** 3.10–3.11
* **Create environment**

  ```bash
  uv venv && source .venv/bin/activate
  uv pip install -r requirements.txt
  ```

* **Core dependencies:** `pandas`, `numpy`, `scikit-learn`, `beautifulsoup4`, `requests`, `tqdm`, `typer`, `jinja2`, `matplotlib`, `transformers`, `torch` (CPU ok), `evaluate`.
* **Pre-commit hooks:** `ruff`, `black`.

---

## 5) Workflow & commit conventions

**Every commit is tagged with the CRISP-DM phase:** `[BU] [DU] [DP] [MO] [EV] [DE] [DOCS]`.
*Example:* `[DP] Segment utterances by speaker; link to motion; save JSONL`
Use a clear, short summary. Use your **UNZA email** for Git/Colab.

**Branches:** `phase/bu`, `phase/du`, … `hotfix/*`.
**PR template:** what/why/how to test.

---

## 6) CRISP-DM checkpoints — what to produce

### \[BU] Business Understanding

* One-paragraph **problem statement**, intended users, risks/assumptions.
* **KPIs:** Macro-F1; Positive (Relevant) Recall; AUPRC.
* **Scope:** sittings covered; languages; ethical notes.
* **Artifacts:** `docs/BU.md`, initial repo scaffold.

### \[DU] Data Understanding

* **Scrape** 6–10 sittings (Debates/Proceedings); store raw HTML/PDF.
* **Collect** corresponding **Order Papers** (motion text); map to sittings.
* **EDA:** utterance length, speakers per sitting, early class prior (from seed labels).
* **Artifacts:** `data/raw/*`, `notebooks/du_eda.ipynb`, `docs/DU.md`.

### \[DP] Data Preparation

* **Segmentation:** `(speaker, timestamp, utterance)` from transcripts.
* **Linkage:** join each utterance to its motion (from Order Paper).
* **Annotation set:** 1,000 utterances labeled (double-annotate 20% for κ ≥ 0.75).
* **Splits:** sitting-wise train/val/test (no leakage).
* **Artifacts:** `data/interim/*.jsonl`, `data/processed/splits/*`, `docs/DP.md`.

### \[MO] Modeling

* **Baselines:** TF-IDF → Linear SVM/LogReg; simple overlap heuristic.
* **Contextual:** cross-encoder (`roberta-base`), pair input: `[CLS] motion [SEP] utterance`.
* **Artifacts:** `experiments/runs/*/metrics.json`, saved models, `docs/MO.md`.

### \[EV] Evaluation

* Macro-F1, Recall (Relevant), AUPRC; per-sitting & per-speaker slices; error taxonomy.
* **Choose operating threshold** for deployment.
* **Artifacts:** `reports/figs/*`, `docs/EV.md`.

### \[DE] Deployment

* **CLI**: `classify_relevance.py --motion motion.txt --transcript sitting.html --out labels.csv`
* **Colab demo**: paste motion + snippet → predictions with confidences.
* **Artifacts:** `src/app/cli.py`, updated Colab, `docs/DE.md`.

---

## 7) Data sources & pipeline

1. **Scrape** National Assembly debates/proceedings (by sitting) → `data/raw/`.
2. **Fetch** the same day’s **Order Paper** → extract motion text → `data/interim/`.
3. **Parse & segment** transcripts into speaker turns; **link** each turn to its motion.
4. **Persist** as JSONL; generate splits and feature stores.

Our repo must include raw & pre-processed data and a reproducible notebook showing extraction → implementation → evaluation → deployment.

---

## 8) Label schema & annotation protocol

* **Labels:** `Relevant`, `NotRelevant`
* **Unit:** one speaker turn (utterance).
* **Context to annotators:** motion text + current utterance (+ optional previous turn).
* **Agreement target:** Cohen’s κ ≥ 0.75 on 20% double-labeled set; adjudicate disagreements.
* **CSV format:** `sitting_id,speaker,timestamp,text,label,annotator,notes`.
* **Quality gates:** banter/procedural points → `NotRelevant`; claims/evidence/proposals tied to motion → `Relevant`.

---

## 9) Modeling plan

* **Baselines:** TF-IDF (word/char n-grams) → Linear SVM/LogReg; rule-based overlap scores.
* **Neural:** cross-encoder (`roberta-base`, max 256–384 tokens per utterance).
* **Imbalance:** class weights or focal loss; probability calibration (Platt or isotonic).
* **Explainability:** feature importances (linear), saliency/attention viz (transformers).

---

## 10) Evaluation

* **Primary:** Macro-F1.
* **Secondary:** Recall (Relevant), AUPRC (Relevant), Balanced Accuracy.
* **Slices:** per-sitting, per-speaker; **Error analysis:** confusion buckets (off-topic jokes, procedural, adjacent topics).

---

## 11) Minimal “demo” deployment

* **CLI usage**

  ```bash
  python -m src.app.cli \
    --motion data/interim/<order_paper>.txt \
    --transcript data/raw/<debate>.html \
    --out experiments/runs/<run_id>/labels.csv
  ```

* **Colab notebook:** end-to-end cells with “Run All” producing metrics + highlighted examples.
* **Slide deck:** 8–12 slides telling the CRISP-DM story; include a demo screenshot.

---

## 12) Roles & ownership (suggested)

* **Data lead:** scraping, parsing, linkage; owns `src/scrape`, `src/parse`.
* **Label lead:** annotation guide, QA, κ checks; owns `src/label`.
* **Model lead:** baselines + transformer; owns `src/models`, `src/features`.
* **Eval lead:** metrics, error analysis, figures; owns `src/eval`, `reports/figs`.
* **Docs/presenter:** notebook polish, paper, slides; owns `docs`, `reports`.
  *All contribute code & commits—graded per activity.*

---

## 13) HOWTO: run the pipeline locally

```bash
# 1) Scrape first sittings and their Order Papers
python -m src.scrape.fetch_sittings --out data/raw/
python -m src.scrape.fetch_order_papers --range <start>:<end> --out data/interim/

# 2) Parse & segment to utterances
python -m src.parse.segment --in data/raw/ --order-papers data/interim/ --out data/interim/utterances.jsonl

# 3) Create labeled subset for DP
python -m src.label.make_seed --in data/interim/utterances.jsonl --n 1000 --out data/processed/seed.csv

# 4) Train baselines
python -m src.models.train_baselines --in data/processed/ --out experiments/runs/baseline_svm/

# 5) Evaluate & plot
python -m src.eval.report --run experiments/runs/baseline_svm/ --out reports/figs/
```

---

## 14) Reporting pack (checklist)

* **Notebook**: reproducible and commented; shows data flow and results.
* **Report**: ACM template, ≤4 pages, self-contained, English.
* **Slides**: PDF, visuals + Q\&A readiness.
* **Repo**: tagged commits across phases by **all** team members.

---

## 15) To-dos by phase (actionable)

**Kickoff (pre-BU & BU)**

* [ ] Create repo with the correct name; add this README.
* [ ] Draft **Problem Statement** + KPIs + risks in `docs/BU.md`.
* [ ] Set up `requirements.txt`, Makefile, CI lint.

**DU**

* [ ] Scrape 6–10 sittings; store raw HTML/PDF.
* [ ] Fetch Order Papers for the same dates; extract motions.
* [ ] EDA notebook with histograms, speaker counts.

**DP**

* [ ] Segment to utterances; link to motions.
* [ ] Label 1,000 utterances; run κ check on 20% sample.
* [ ] Freeze splits (sitting-wise).

**MO**

* [ ] Train SVM/LogReg baselines; log `metrics.json`.
* [ ] Train RoBERTa cross-encoder; compare AUPRC/Recall.
* [ ] Calibrate probabilities; pick threshold.

**EV**

* [ ] Produce slice metrics tables & confusion matrix.
* [ ] Error taxonomy & 6 qualitative examples for the report.

**DE**

* [ ] CLI + Colab demo finalized.
* [ ] Freeze artifacts; update notebook “Run All”.
* [ ] Submit report, repo link, Colab, slides (exact filenames).
