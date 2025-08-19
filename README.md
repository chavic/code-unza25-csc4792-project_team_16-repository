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

* **Python:** 3.10–3.11 (tested with 3.11.13)
* **Create environment**

  ```bash
  # Option 1: Using conda (recommended)
  conda create -n parliament-classifier python=3.11 -y
  conda activate parliament-classifier
  pip install -r requirements.txt
  
  # Option 2: Using uv
  uv venv && source .venv/bin/activate
  uv pip install -r requirements.txt
  ```

* **Core dependencies:** `pandas`, `numpy`, `scikit-learn`, `beautifulsoup4`, `requests`, `tqdm`, `typer`, `jinja2`, `matplotlib`, `transformers`, `torch` (CPU ok), `evaluate`, `urllib3`.
* **Pre-commit hooks:** `ruff`, `black`.

### 🚨 **SSL Certificate Note**
The Parliament website has SSL certificate issues. The scrapers handle this automatically by disabling SSL verification for the parliament.gov.zm domain only.

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

### **Data Sources (Zambian National Assembly)**
- **Main debates index:** `https://www.parliament.gov.zm/publications/debates-list`
- **Alternate debates index:** `https://www.parliament.gov.zm/publications/debates-proceedings`
- **Order papers index:** `https://www.parliament.gov.zm/publications/order-paper-list`
- **Votes & proceedings:** `https://www.parliament.gov.zm/publications/votes-proceedings` (validation)

### **Pipeline**
1. **Scrape** National Assembly debates/proceedings (by sitting) → `data/raw/`.
2. **Fetch** the same day's **Order Paper** → extract motion text → `data/interim/`.
3. **Parse & segment** transcripts into speaker turns; **link** each turn to its motion.
4. **Persist** as JSONL; generate splits and feature stores.

### **🎯 Complete Dataset Scraped (50.6MB)**
- **199 parliamentary debates** (43MB) with full transcripts and speaker attribution
- **200 order papers** (7.6MB) with motion text and session information
- **Date range:** 2022-2025 with comprehensive historical coverage
- **Success rate:** 99.5% for debates, 100% for order papers

Our repo includes raw & pre-processed data and a reproducible notebook showing extraction → implementation → evaluation → deployment.

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

### **🚀 Quick Start: Complete Data Scraping**

```bash
# Activate environment
conda activate parliament-classifier  # or source .venv/bin/activate

# 1) Scrape ALL parliamentary debates (comprehensive dataset)
python -m src.scrape.fetch_sittings \
  --out data/raw \
  --num-sittings 200 \
  --delay 1.5 \
  --max-pages 15 \
  --use-alt-index

# 2) Scrape ALL order papers (motion texts)
python -m src.scrape.fetch_order_papers \
  --out data/interim \
  --max-papers 200 \
  --delay 1.5 \
  --max-pages 15

# 3) Parse & segment to utterances
python -m src.parse.segment --in data/raw/ --order-papers data/interim/ --out data/interim/utterances.jsonl

# 4) Create labeled subset for DP
python -m src.label.make_seed --in data/interim/utterances.jsonl --n 1000 --out data/processed/seed.csv

# 5) Train baselines
python -m src.models.train_baselines --in data/processed/ --out experiments/runs/baseline_svm/

# 6) Evaluate & plot
python -m src.eval.report --run experiments/runs/baseline_svm/ --out reports/figs/
```

### **⚡ Quick Test: Small Dataset**

```bash
# Test with just a few sittings (for development/testing)
python -m src.scrape.fetch_sittings --out data/raw --num-sittings 5 --delay 2.0 --max-pages 1
python -m src.scrape.fetch_order_papers --out data/interim --max-papers 5 --delay 2.0 --max-pages 1
```

### **📊 Scraper Options**

**Debates Scraper (`fetch_sittings`):**
- `--num-sittings`: Number of debate transcripts to scrape (default: 10)
- `--delay`: Delay between requests in seconds (default: 1.0, recommended: 1.5-2.0)
- `--max-pages`: Maximum index pages to crawl (default: 5, full site: 15)
- `--use-alt-index`: Also scrape alternate debates index for broader coverage
- `--start-year`: Starting year filter (default: 2023)

**Order Papers Scraper (`fetch_order_papers`):**
- `--max-papers`: Number of order papers to scrape (default: 100)
- `--delay`: Delay between requests in seconds (default: 1.0)
- `--max-pages`: Maximum index pages to crawl (default: 10, full site: 15)

### **📁 Expected Output Structure**

```
data/
├── raw/ (43MB after full scrape)
│   ├── sitting_2025-07-15_12493_478402d6.html  # Full debate transcript
│   ├── sitting_2025-07-15_12493_478402d6.json  # Metadata (URL, date, hash)
│   └── ... (199 more sittings)
└── interim/ (7.6MB after full scrape)
    ├── order_paper_2025-07-25_12469_eb873982.json  # Motion data + session info
    ├── order_paper_2025-07-25_12469_eb873982.html  # Raw HTML
    └── ... (200 more order papers)
```

### **🔍 Data Quality Verification**

```bash
# Check scraping results
ls data/raw/ | grep sitting | wc -l    # Should show ~400 files (HTML + JSON)
ls data/interim/ | grep order_paper | wc -l  # Should show ~400 files
du -sh data/raw/ data/interim/         # Check total data size

# Examine sample files
head -20 data/raw/sitting_*.json       # Check metadata structure
head -50 data/interim/order_paper_*.json  # Check motion extraction
```

### **🛠️ Troubleshooting**

**SSL Certificate Errors:**
```bash
# The scrapers automatically handle SSL issues, but if you see certificate errors:
# - Check internet connection
# - Verify the parliament.gov.zm website is accessible
# - The scrapers use verify=False for parliament.gov.zm only
```

**Timeout Errors:**
```bash
# If you get timeout errors:
# - Increase the --delay parameter (try 2.0 or 3.0)
# - Reduce --num-sittings for testing
# - Check your internet connection stability
```

**Memory Issues:**
```bash
# For large scrapes, monitor memory usage:
htop  # or Activity Monitor on macOS
# The scrapers process one file at a time, so memory should be stable
```

**Partial Scraping Results:**
```bash
# If scraping stops early:
# - Check the terminal output for specific error messages
# - Restart with a smaller batch size
# - The scrapers support resuming (won't re-download existing files)
```

### **📈 Performance Notes**

- **Full scraping time:** ~15 minutes (199 debates + 200 order papers)
- **Network requirements:** Stable internet connection
- **Disk space:** ~51MB for complete dataset
- **Politeness:** Built-in delays respect server resources
- **Deduplication:** Content hashing prevents re-downloading

### **📋 Data Documentation**

#### **Debate Files Structure (`data/raw/`)**
```json
{
  "url": "https://www.parliament.gov.zm/node/12493",
  "node_id": "12493", 
  "title": "Tuesday, 15th July, 2025",
  "date": "2025-07-15",
  "source_index": "https://www.parliament.gov.zm/publications/debates-list",
  "content_hash": "478402d6",
  "scraped_at": "2025-08-19T14:13:10.971712",
  "filename": "sitting_2025-07-15_12493_478402d6.html"
}
```

#### **Order Paper Files Structure (`data/interim/`)**
```json
{
  "url": "https://www.parliament.gov.zm/node/12469",
  "node_id": "12469",
  "title": "Friday, 25th July, 2025", 
  "date": "2025-07-25",
  "scraped_at": "2025-08-19T14:36:27.206378",
  "motions": [
    {
      "motion_id": "M_001",
      "text": "That this House...",
      "type": "substantive",
      "mover": "Hon. Member Name"
    }
  ],
  "order_items": [
    {
      "item_number": 1,
      "text": "Item description",
      "type": "motion"
    }
  ],
  "session_info": {
    "session": "FOURTH Session",
    "assembly": "THIRTEENTH Assembly", 
    "sitting_date": "FRIDAY, 25TH JULY, 2025"
  }
}
```

#### **File Naming Convention**
- **Debates:** `sitting_{date}_{node_id}_{hash}.{html|json}`
- **Order Papers:** `order_paper_{date}_{node_id}_{hash}.{html|json}`
- **Date format:** `YYYY-MM-DD` (e.g., `2025-07-15`)
- **Content hash:** First 8 characters of MD5 hash for deduplication

#### **🚨 Important: Data Not in Git**
The scraped data files (50.6MB total) are excluded from git via `.gitignore` to keep the repository size manageable. To replicate the full dataset:

1. Clone the repository
2. Set up the environment (see Setup section)
3. Run the scraping commands (see Quick Start section)
4. The scrapers will recreate the exact same dataset structure

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
