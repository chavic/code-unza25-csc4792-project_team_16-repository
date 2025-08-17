# Evaluation (EV)

**Phase:** [EV] Evaluation  
**Date:** 2025  
**Team:** Team 16  

## Overview

This phase provides comprehensive evaluation of trained models, including performance analysis, error taxonomy, and model selection for deployment.

## Evaluation Framework

### Primary Evaluation Metrics

1. **Macro-F1 Score**
   - Balances performance across both classes (Relevant/NotRelevant)
   - Primary metric for model comparison
   - Target: ≥ 0.75

2. **Relevant Class Recall**
   - Captures the proportion of truly relevant utterances identified
   - Critical for parliamentary indexing use case
   - Target: ≥ 0.80

3. **Area Under Precision-Recall Curve (AUPRC)**
   - Handles class imbalance better than ROC-AUC
   - Focuses on positive class performance
   - Target: ≥ 0.70

### Secondary Metrics

- **Balanced Accuracy**: Account for class imbalance
- **Precision (Relevant)**: Quality of positive predictions
- **Specificity**: True negative rate for NotRelevant class
- **Cohen's Kappa**: Agreement beyond chance

## Evaluation Pipeline

### Model Performance Assessment
```bash
python -m src.eval.report --run experiments/runs/baseline_svm/ --out reports/figs/
python -m src.eval.report --run experiments/runs/roberta_cross_encoder/ --out reports/figs/
```

### Comprehensive Analysis

1. **Overall Performance**
   - Test set evaluation for all trained models
   - Statistical significance testing between models
   - Confidence intervals for key metrics

2. **Slice Analysis**
   - Per-sitting performance variation
   - Per-speaker bias detection
   - Per-motion-type performance differences
   - Utterance length impact analysis

3. **Threshold Optimization**
   - Precision-recall curve analysis
   - Operating point selection for deployment
   - Cost-benefit analysis for different thresholds

## Error Analysis

### Error Taxonomy

**Type 1: False Positives (NotRelevant → Relevant)**
- **Procedural confusion**: Procedural statements misclassified as substantive
- **Tangential references**: Off-topic mentions of motion keywords
- **Contextual misunderstanding**: Requires broader context to classify correctly

**Type 2: False Negatives (Relevant → NotRelevant)**  
- **Implicit relevance**: Relevant without explicit motion keywords
- **Complex arguments**: Multi-step reasoning not captured
- **Speaker style**: Indirect or metaphorical speech patterns

**Type 3: Annotation Disagreement**
- **Ambiguous cases**: Genuinely difficult to classify
- **Annotator inconsistency**: Different interpretation of guidelines
- **Context dependency**: Requires previous utterances for clarity

### Qualitative Analysis

**Sample Error Cases** (6 examples for technical report):
1. **False Positive Example**: Procedural statement with motion keywords
2. **False Negative Example**: Implicit support without direct reference
3. **Ambiguous Case**: Borderline relevance requiring judgment
4. **Context-Dependent**: Needs previous utterance for classification
5. **Speaker Style**: Indirect communication pattern
6. **Complex Argument**: Multi-part reasoning chain

### Error Impact Assessment
- **High-impact errors**: Clearly relevant/irrelevant cases misclassified
- **Low-impact errors**: Borderline cases with reasonable disagreement
- **Systematic biases**: Consistent errors across specific patterns

## Slice Performance Analysis

### Per-Sitting Analysis
```python
# Evaluate performance variation across parliamentary sittings
sitting_metrics = {}
for sitting_id in test_sittings:
    sitting_data = test_df[test_df['sitting_id'] == sitting_id]
    metrics = evaluate_model(model, sitting_data)
    sitting_metrics[sitting_id] = metrics
```

**Analysis Questions**:
- Do some sittings consistently perform better/worse?
- Are there topic-specific performance patterns?
- How does sitting length affect classification quality?

### Per-Speaker Analysis
```python
# Detect potential speaker bias
speaker_metrics = {}
for speaker in frequent_speakers:
    speaker_data = test_df[test_df['speaker'] == speaker]
    metrics = evaluate_model(model, speaker_data)
    speaker_metrics[speaker] = metrics
```

**Bias Detection**:
- Are some speakers consistently misclassified?
- Does model performance correlate with speaker characteristics?
- Are there systematic biases to address?

### Per-Motion-Type Analysis
- **Economic motions**: Budget, taxation, development
- **Social motions**: Health, education, welfare
- **Procedural motions**: Rules, appointments, procedures
- **International motions**: Foreign relations, treaties

## Statistical Analysis

### Model Comparison
- **Paired t-tests**: Compare model performance on same test cases
- **McNemar's test**: Assess classification differences between models
- **Bootstrap confidence intervals**: Estimate metric uncertainty

### Performance Significance
- **Effect size**: Practical significance of performance differences
- **Power analysis**: Sample size adequacy for conclusions
- **Cross-validation**: Stability of results across data splits

## Visualization and Reporting

### Performance Visualizations

1. **Confusion Matrix Heatmaps**
   ```python
   sns.heatmap(confusion_matrix, annot=True, fmt='d', 
               xticklabels=['NotRelevant', 'Relevant'],
               yticklabels=['NotRelevant', 'Relevant'])
   ```

2. **Precision-Recall Curves**
   ```python
   plt.plot(recall, precision, label=f'{model_name} (AUPRC={auprc:.3f})')
   ```

3. **Metrics Comparison Charts**
   - Bar plots comparing F1, precision, recall across models
   - Box plots showing slice performance variation
   - Scatter plots for error pattern analysis

### Performance Tables
```
| Model              | Macro-F1 | Relevant Recall | AUPRC | Balanced Acc |
|--------------------|----------|-----------------|-------|--------------|
| Logistic Regression| 0.68     | 0.75           | 0.72  | 0.69         |
| SVM                | 0.71     | 0.78           | 0.74  | 0.72         |
| RoBERTa            | 0.79     | 0.85           | 0.82  | 0.80         |
```

## Threshold Selection

### Operating Point Analysis
- **High Precision**: Minimize false positives (conservative classification)
- **High Recall**: Minimize false negatives (comprehensive capture)
- **Balanced**: Optimize F1 score for general use
- **Custom**: Domain-specific cost considerations

### Deployment Threshold
**Recommended approach**: Optimize for Relevant class recall while maintaining reasonable precision
- Target: Recall ≥ 0.80, Precision ≥ 0.70
- Rationale: Better to over-include than miss relevant utterances
- Validation: Test on held-out validation set

## Model Selection

### Selection Criteria
1. **Primary metrics**: Macro-F1, Relevant Recall, AUPRC
2. **Generalization**: Consistent performance across slices
3. **Interpretability**: Ability to explain predictions
4. **Computational efficiency**: Inference speed and resource requirements
5. **Robustness**: Performance on edge cases and noisy data

### Recommendation Framework
```python
def select_best_model(model_results):
    # Weight different criteria
    scores = {}
    for model in model_results:
        score = (
            0.4 * model['macro_f1'] +
            0.3 * model['relevant_recall'] +
            0.2 * model['auprc'] +
            0.1 * model['generalization_score']
        )
        scores[model['name']] = score
    return max(scores, key=scores.get)
```

## Deliverables

### Quantitative Results
- **Performance metrics table**: All models on test set
- **Statistical significance tests**: Model comparison results
- **Slice analysis tables**: Performance breakdown by sitting/speaker/motion
- **Confusion matrices**: Error pattern visualization

### Qualitative Analysis
- **Error taxonomy**: Categorized failure modes
- **Example cases**: Representative errors with explanations
- **Bias assessment**: Systematic performance issues
- **Improvement recommendations**: Specific suggestions for next iteration

### Visualizations
- **Performance comparison charts**: Model metrics side-by-side
- **Precision-recall curves**: Threshold selection visualization
- **Slice performance plots**: Variation across data subsets
- **Error distribution analysis**: Pattern identification

### Model Selection Report
- **Recommended model**: Best performer with justification
- **Operating threshold**: Optimal classification threshold
- **Expected performance**: Projected real-world metrics
- **Deployment considerations**: Practical implementation notes

## Success Criteria

This phase succeeds when:
1. **Comprehensive evaluation**: All models assessed on multiple metrics
2. **Error understanding**: Clear taxonomy of failure modes
3. **Slice analysis**: Performance variation documented and explained
4. **Model selection**: Clear recommendation with justification
5. **Deployment readiness**: Threshold and configuration determined

---

**Next Phase:** [DE] Deployment - CLI tool and demonstration system
