# Data Understanding (DU)

**Phase:** [DU] Data Understanding  
**Date:** 2025  
**Team:** Team 16  

## Data Collection Overview

This phase focuses on gathering and understanding the raw data needed for the parliament relevance classification task.

## Data Sources

### 1. Parliamentary Sitting Transcripts

- **Source**: Zambian National Assembly website
- **Format**: HTML pages with structured debate content
- **Target**: 6-10 sittings from 2023
- **Content**: Speaker attributions, timestamps, utterance text
- **Storage**: `data/raw/*.html`

### 2. Order Papers

- **Source**: Corresponding Order Papers for each sitting date
- **Format**: PDF or HTML documents
- **Content**: Motion text, motion IDs, motion movers
- **Purpose**: Ground truth for what motions were being debated
- **Storage**: `data/interim/*.json` (extracted motion text)

## Data Collection Process

### Scraping Workflow

1. **Identify sitting dates** - Select representative sample from 2023
2. **Download transcripts** - Scrape HTML content with speaker turns
3. **Fetch Order Papers** - Download corresponding motion documents
4. **Extract motion text** - Parse structured motion information
5. **Quality checks** - Verify data completeness and format

### Technical Implementation

```bash
# Scrape parliamentary sittings
python -m src.scrape.fetch_sittings --out data/raw/

# Fetch corresponding Order Papers  
python -m src.scrape.fetch_order_papers --range 2023-01:2023-12 --out data/interim/
```

## Exploratory Data Analysis (EDA)

### Key Questions to Answer

1. How many utterances per sitting on average?
2. What is the distribution of utterance lengths?
3. How many unique speakers participate?
4. What types of motions are most common?
5. What is the preliminary class distribution (relevant vs not relevant)?

### Expected Findings

- **Utterance length**: Expect wide variation from short procedural statements to long policy arguments
- **Speaker participation**: Some speakers likely much more active than others
- **Motion types**: Mix of economic, social, and procedural motions
- **Class imbalance**: Likely more relevant than not relevant utterances

### EDA Notebook Structure

The analysis will be documented in `notebooks/du_eda.ipynb` covering:

1. **Data Loading and Basic Stats**
   - Number of sittings, utterances, speakers
   - Date range coverage
   - File sizes and formats

2. **Text Analysis**
   - Utterance length distributions
   - Most common words and phrases
   - Speaker vocabulary diversity

3. **Motion Analysis**
   - Motion types and categories
   - Motion text length and complexity
   - Temporal patterns

4. **Quality Assessment**
   - Missing data identification
   - Transcript quality indicators
   - Motion-sitting linkage success rate

## Data Quality Considerations

### Potential Issues

- **Incomplete transcripts** - Some utterances may be truncated
- **Speaker misattribution** - Names might be inconsistent
- **Timestamp accuracy** - May be approximate rather than precise
- **Motion linkage** - Order Papers may not perfectly align with debate content

### Quality Metrics

- Percentage of utterances with clear speaker attribution
- Percentage of sittings with corresponding Order Papers
- Average utterance completeness (no truncation markers)
- Motion-sitting temporal alignment accuracy

## Initial Data Schema

### Utterance Record Structure

```json
{
  "sitting_id": "sitting_2023_001",
  "utterance_id": "sitting_2023_001_042",
  "speaker": "Hon. Member Name",
  "timestamp": "2023-03-15T14:30:00",
  "text": "Mr. Speaker, I rise to support this important motion...",
  "motion_id": "M2023_001",
  "motion_text": "That this House calls upon..."
}
```

### Motion Record Structure

```json
{
  "motion_id": "M2023_001",
  "title": "Motion on Economic Development",
  "text": "That this House calls upon the Government...",
  "mover": "Hon. Member A",
  "date": "2023-03-15",
  "type": "substantive"
}
```

## Expected Outcomes

By the end of this phase, we will have:

1. **Raw Data Collection** - 6-10 sitting transcripts and corresponding Order Papers
2. **Data Understanding** - Clear picture of data structure, quality, and characteristics
3. **EDA Documentation** - Comprehensive analysis in Jupyter notebook
4. **Data Issues Identification** - Known limitations and quality concerns
5. **Processing Strategy** - Plan for data preparation phase based on findings

## Risks Identified

- **Website structure changes** - Scraping may break if site is updated
- **Access restrictions** - Rate limiting or blocking of automated requests
- **Data completeness** - Some sittings may have incomplete or missing transcripts
- **Format inconsistencies** - Different sittings may use different HTML structures

## Next Steps

The findings from this phase will inform the data preparation strategy, particularly:

- Text preprocessing requirements
- Speaker name standardization needs  
- Motion linkage algorithm design
- Annotation sampling strategy

---

**Next Phase:** [DP] Data Preparation - Segmentation, linkage, and annotation
