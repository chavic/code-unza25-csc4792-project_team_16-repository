# Data Understanding (DU)

**Phase:** [DU] Data Understanding  
**Date:** 2025  
**Team:** Team 16

## 2.1 Sources & provenance

- **Debates & Proceedings (index):** public list of House debates by sitting; this is our primary transcript source. ([Parliament of Zambia][1])
- **Order Papers (index):** agenda for each sitting; provides the **motion text** we will condition on. ([Parliament of Zambia][2])
- **Votes & Proceedings (index):** timing/outcomes; helps validate date/session alignment across pages. ([Parliament of Zambia][3])
- **Site structure (top-level):** the National Assembly homepage exposes both the current “Debates and Proceedings” and an **(OLD)** archive—useful when crawling older sessions. ([Parliament of Zambia][4])
- **Verbatim status:** Parliament’s publications confirm Daily Parliamentary Debates (Hansard) is the official verbatim record; one procedural abstract states it is “essentially a verbatim account”. This justifies text-only modeling as a sound first pass. ([Parliament of Zambia][5])
- **Custody:** Clerk’s Office is the custodian of records (minutes → Votes & Proceedings), reinforcing provenance. ([Parliament of Zambia][6])

## 2.2 What’s on a sitting page (expected fields)

Typical debate pages (by date) include a **session header** and the **verbatim debate text** (e.g., “Daily Parliamentary Debates … The House met at 1430 hours”), from which we’ll segment **speaker turns**. ([Parliament of Zambia][7])  
Order Paper pages are date-keyed and list the **Order of the Day**, from which we’ll extract the **motion**. ([Parliament of Zambia][8])

**Target raw fields to harvest**

- From **Debates**: `date`, `assembly/session`, `sitting_title`, `full_text`, implicit `speaker` delimiters (turn headings), stage cues (e.g., “Point of Order”). ([Parliament of Zambia][7])
- From **Order Paper**: `date`, `order_items[]`, **`motion_text`**. ([Parliament of Zambia][8])
- From **Votes & Proceedings**: `date`, `sitting_meta`, `outcomes`. ([Parliament of Zambia][3])

## 2.3 Coverage & format realities

- **Multiple catalogs:** current “Debates & Proceedings” plus “(OLD)” archive—plan for **two index patterns**. ([Parliament of Zambia][4])
- **Mixed formats:** Some content is HTML node pages; certain documents (procedural abstracts, standing orders) are PDFs—our crawler must support both. ([Parliament of Zambia][9])

## 2.4 Join strategy (date/session keyed)

For a given **date**:

1. pull **Order Paper** → extract **motion**;
2. pull **Debates** → segment into `(speaker, timestamp?, utterance)`;
3. (optional) cross-check date/session via **Votes & Proceedings**. ([Parliament of Zambia][2])

## 2.5 Risks & data-quality notes

| Risk                            | Why it matters                  | DU action                                                                                                                                                                         |
| ------------------------------- | ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Template drift (old vs new)** | Parser breaks on older sessions | Crawl both catalogs; versioned parsers per template. ([Parliament of Zambia][4])                                                                                                  |
| **HTML/PDF variance**           | Some docs only as PDFs          | Add PDF text extraction path; keep raw snapshots. ([Parliament of Zambia][9])                                                                                                     |
| **Ambiguous “relevance” edges** | Label noise                     | Start an annotation guide with concrete on/off-topic examples from Zambia sittings; double-label a subset for κ. (Verbatim status supports fidelity.) ([Parliament of Zambia][5]) |
| **Session/date mismatches**     | Fragile joins                   | Use **date** as primary key; validate against Votes & Proceedings. ([Parliament of Zambia][3])                                                                                    |

## 2.6 Data dictionary (processed layer)

We’ll normalize into **utterance-level** rows:

| Field              | Type    | Source      | Notes                                                                                  |
| ------------------ | ------- | ----------- | -------------------------------------------------------------------------------------- |
| `sitting_id`       | string  | Debates     | e.g., `2025-06-25` (date key). ([Parliament of Zambia][1])                             |
| `assembly_session` | string  | Debates     | “Fourth Session of the Thirteenth Assembly”, when present. ([Parliament of Zambia][8]) |
| `speaker`          | string  | Debates     | Parsed from turn headings. ([Parliament of Zambia][7])                                 |
| `timestamp`        | string? | Debates     | If present (“The House met at 1430 hours”). ([Parliament of Zambia][7])                |
| `utterance_text`   | text    | Debates     | Cleaned verbatim text. ([Parliament of Zambia][7])                                     |
| `stage_marker`     | enum    | Debates     | e.g., `POINT_OF_ORDER`, `INTERJECTION`.                                                |
| `motion_text`      | text    | Order Paper | Target text for conditioning. ([Parliament of Zambia][8])                              |
| `label`            | enum    | Annotation  | `Relevant`/`NotRelevant`                                                               |
| `split`            | enum    | Processing  | `train`/`val`/`test` (sitting-wise)                                                    |

## 2.7 DU tasks & quick EDA

**DU-01: Crawl small seed set**

- From the **Debates & Proceedings** index, collect 3–5 recent sittings; store raw HTML with content hashes. ([Parliament of Zambia][1])
- For the same dates, fetch **Order Papers**; store raw and a text-extracted `motion.txt`. ([Parliament of Zambia][2])

**DU-02: Parse & segment prototype**

- Regex/DOM-based segmentation of speaker turns from one sitting page (verify on at least one older sitting too). ([Parliament of Zambia][7])

**DU-03: Sanity EDA** (on segmented utterances)

- Histograms: utterance length (tokens), per-speaker turn counts, `%` of stage markers (e.g., Point of Order).
- Early class prior (using a **lexical-overlap heuristic** with the motion to get **seed labels** for inspection).

**DU-04: Data card**

- One-page “data card” documenting sources, date of access, scraping rules, known quirks, and contact—linking the Clerk’s record role for provenance context. ([Parliament of Zambia][6])

## 2.8 Ethics & compliance

- We will not use **speaker identity** as a model feature (only for evaluation slices), mitigating shortcut learning and fairness concerns.
- Keep raw snapshots and extraction scripts in-repo for **reproducibility**.

---

## 2.9 Ready-to-run TODOs (copy into your issue tracker)

- [ ] [DU] Seed crawl: 3–5 sittings from **Debates & Proceedings** + matching **Order Papers**; save under `data/raw/`. ([Parliament of Zambia][1])
- [ ] [DU] Write `parse_segment.py` to produce `data/interim/utterances.jsonl` with `(speaker, utterance_text, stage_marker)` from one sitting. ([Parliament of Zambia][7])
- [ ] [DU] Extract `motion_text` from Order Papers into `data/interim/<date>_motion.txt`. ([Parliament of Zambia][8])
- [ ] [DU] EDA notebook: length hist, turns per speaker, marker frequencies; attach screenshots of the source index in `docs/moodle/du/`. ([Parliament of Zambia][1])
- [ ] [DU] Data card v0.1 with provenance & verbatim references. ([Parliament of Zambia][5])

[1]: https://www.parliament.gov.zm/publications/debates-list?utm_source=chatgpt.com "Debates and Proceedings | National Assembly of Zambia"
[2]: https://www.parliament.gov.zm/publications/order-paper-list?utm_source=chatgpt.com "Order Paper | National Assembly of Zambia"
[3]: https://www.parliament.gov.zm/publications/votes-proceedings?utm_source=chatgpt.com "Votes and Proceedings | National Assembly of Zambia"
[4]: https://www.parliament.gov.zm/?utm_source=chatgpt.com "National Assembly of Zambia"
[5]: https://www.parliament.gov.zm/node/173?utm_source=chatgpt.com "Publications | National Assembly of Zambia"
[6]: https://www.parliament.gov.zm/the-clerk?utm_source=chatgpt.com "The Clerk's Office"
[7]: https://www.parliament.gov.zm/node/1401?utm_source=chatgpt.com "Debates- Thursday, 4th November, 2010"
[8]: https://www.parliament.gov.zm/node/12397?utm_source=chatgpt.com "Wednesday, 25th June, 2025 | National Assembly of Zambia"
[9]: https://www.parliament.gov.zm/sites/default/files/images/publication_docs/Abstract%202%20Debate%20In%20Parliament.pdf?utm_source=chatgpt.com "Abstract 2 Debate In Parliament.pdf"

---

**Next Phase:** [DP] Data Preparation - Segmentation, linkage, and annotation
