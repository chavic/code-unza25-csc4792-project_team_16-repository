# Data Preparation (DP)

**Phase:** [DP] Data Preparation  
**Date:** 2025  
**Team:** Team 16  

## Overview

This phase transforms raw parliamentary data into a clean, labeled dataset ready for machine learning model training.

## Data Processing Pipeline

### 1. Transcript Segmentation
**Goal**: Convert raw HTML transcripts into structured utterances

**Process**:
```bash
python -m src.parse.segment --in data/raw/ --order-papers data/interim/ --out data/interim/utterances.jsonl
```

**Tasks**:
- Parse HTML structure to identify speaker turns
- Extract speaker names, timestamps, and utterance text
- Handle formatting inconsistencies and special cases
- Link each utterance to its corresponding motion from Order Papers

**Output**: `data/interim/utterances.jsonl` with structured utterance records

### 2. Motion Linkage
**Goal**: Associate each utterance with the relevant motion being debated

**Approach**:
- **Temporal matching**: Link utterances to motions based on sitting date
- **Keyword overlap**: Use text similarity between utterance and motion
- **Parliamentary procedure**: Consider motion introduction and debate order
- **Manual validation**: Spot-check linkage accuracy on sample

**Quality Metrics**:
- Percentage of utterances successfully linked to motions
- Accuracy of linkage on manually validated sample
- Distribution of utterances per motion

### 3. Annotation Dataset Creation

#### Sampling Strategy
**Goal**: Create balanced, representative sample for manual labeling

```bash
python -m src.label.make_seed --in data/interim/utterances.jsonl --n 1000 --out data/processed/seed.csv
```

**Sampling Criteria**:
- **Stratified by sitting**: Ensure representation across all parliamentary sessions
- **Speaker diversity**: Include utterances from different speakers
- **Length variety**: Mix of short and long utterances
- **Motion coverage**: Represent different types of motions

**Target**: 1,000 utterances for annotation

#### Annotation Protocol

**Labels**: Binary classification
- **Relevant**: Utterances that directly address the motion (arguments, evidence, proposals)
- **NotRelevant**: Procedural points, greetings, tangents, administrative matters

**Annotation Guidelines**:
1. **Context provided**: Motion text + current utterance (+ optional previous turn)
2. **Clear examples**: Provide positive and negative examples for each category
3. **Edge cases**: Guidelines for handling ambiguous cases
4. **Consistency checks**: Regular calibration sessions between annotators

**Quality Assurance**:
- **Double annotation**: 20% of utterances labeled by two annotators
- **Inter-annotator agreement**: Target Cohen's κ ≥ 0.75
- **Adjudication process**: Resolve disagreements through discussion
- **Annotation validation**: Regular spot checks and feedback

#### Annotation Format
```csv
sitting_id,utterance_id,speaker,timestamp,text,motion_id,motion_text,label,annotator,notes,confidence
```

**Required fields**:
- `label`: "Relevant" or "NotRelevant"
- `annotator`: Annotator identifier
- `confidence`: 1-5 scale (annotator confidence)
- `notes`: Optional comments for difficult cases

### 4. Data Splits Creation

**Goal**: Create train/validation/test splits with no data leakage

**Strategy**: **Sitting-wise splitting** to prevent leakage
- Training utterances and test utterances must come from different sittings
- Ensures model generalizes to new parliamentary sessions
- Maintains temporal and contextual independence

**Split Proportions**:
- **Training**: 60% of sittings (~600 utterances)
- **Validation**: 20% of sittings (~200 utterances)  
- **Test**: 20% of sittings (~200 utterances)

**Output Structure**:
```
data/processed/splits/
├── train.csv
├── val.csv
└── test.csv
```

## Data Quality Metrics

### Pre-Processing Quality
- **Segmentation accuracy**: % of utterances properly extracted
- **Speaker attribution**: % with correct speaker identification
- **Timestamp consistency**: Temporal ordering validation
- **Text completeness**: No truncated or corrupted utterances

### Post-Processing Quality
- **Motion linkage success**: % of utterances linked to motions
- **Annotation completeness**: % of sampled utterances labeled
- **Inter-annotator agreement**: Cohen's κ on double-labeled subset
- **Class balance**: Distribution of Relevant vs NotRelevant labels

## Expected Challenges and Solutions

### Challenge 1: Inconsistent HTML Structure
**Problem**: Different sittings may have different HTML formats
**Solution**: Robust parsing with fallback strategies and manual validation

### Challenge 2: Speaker Name Variations
**Problem**: Same speaker may appear with different name formats
**Solution**: Name standardization and fuzzy matching algorithms

### Challenge 3: Motion Linkage Ambiguity  
**Problem**: Multiple motions discussed in single sitting
**Solution**: Temporal ordering + keyword matching + manual validation

### Challenge 4: Annotation Consistency
**Problem**: Subjective nature of relevance classification
**Solution**: Detailed guidelines, training sessions, and regular calibration

## Deliverables

### Primary Outputs
1. **Segmented utterances**: `data/interim/utterances.jsonl`
2. **Annotated dataset**: `data/processed/seed.csv`
3. **Train/val/test splits**: `data/processed/splits/*.csv`
4. **Quality report**: Inter-annotator agreement and data statistics

### Documentation
1. **Processing pipeline**: Code documentation and usage examples
2. **Annotation guidelines**: Detailed labeling instructions
3. **Quality assessment**: Data quality metrics and validation results
4. **Known limitations**: Identified issues and potential impacts

## Validation Steps

### Technical Validation
- [ ] All utterances have required fields populated
- [ ] No duplicate utterance IDs across dataset
- [ ] Temporal consistency in timestamps
- [ ] Motion linkage quality on sample validation

### Annotation Validation
- [ ] Inter-annotator agreement κ ≥ 0.75 achieved
- [ ] All sampled utterances have labels
- [ ] Class distribution is reasonable (not extreme imbalance)
- [ ] Annotation guidelines followed consistently

### Split Validation
- [ ] No sitting appears in multiple splits
- [ ] Split sizes approximately match target proportions
- [ ] Class distribution similar across splits
- [ ] No data leakage between train/val/test

## Success Criteria

This phase is successful when:
1. **Complete pipeline**: Raw data → processed → annotated → splits
2. **Quality thresholds met**: κ ≥ 0.75, >95% successful processing
3. **Ready for modeling**: Clean train/val/test sets with consistent format
4. **Documentation complete**: Clear process documentation and guidelines

---

**Next Phase:** [MO] Modeling - Baseline and transformer model training
