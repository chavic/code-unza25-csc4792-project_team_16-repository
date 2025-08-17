# Modeling (MO)

**Phase:** [MO] Modeling  
**Date:** 2025  
**Team:** Team 16  

## Overview

This phase develops and trains machine learning models to classify parliamentary utterances as relevant or not relevant to motions.

## Modeling Strategy

### Two-Tier Approach
1. **Baseline models** - Establish performance floor with traditional ML
2. **Advanced models** - Leverage transformer architecture for improved performance

### Input Representation
**Feature format**: Combined motion and utterance text
```
"[MOTION] That this House calls upon... [SEP] [UTTERANCE] Mr. Speaker, I rise to support..."
```

This approach allows models to learn relevance patterns between motion content and speaker responses.

## Baseline Models

### 1. TF-IDF + Linear Models

**Feature Engineering**:
```python
TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),          # Unigrams and bigrams
    stop_words='english',
    min_df=2,                    # Remove very rare terms
    max_df=0.95                  # Remove very common terms
)
```

**Models**:
- **Logistic Regression**: Fast, interpretable baseline
- **Linear SVM**: Alternative linear classifier with different decision boundary

**Training**:
```bash
python -m src.models.train_baselines --in data/processed/ --out experiments/runs/baseline_svm/
```

### 2. Rule-Based Baseline
**Simple heuristic**: Word overlap between motion and utterance
- Calculate Jaccard similarity between motion and utterance word sets
- Classify as Relevant if similarity > threshold
- Provides interpretable baseline for comparison

## Advanced Models

### RoBERTa Cross-Encoder

**Architecture**: Fine-tuned RoBERTa for sequence pair classification
- **Base model**: `roberta-base` (125M parameters)
- **Input format**: `[CLS] motion_text [SEP] utterance_text [SEP]`
- **Output**: Binary classification (Relevant/NotRelevant)
- **Max sequence length**: 384 tokens (accommodate both texts)

**Training Configuration**:
```python
TrainingArguments(
    learning_rate=2e-5,
    batch_size=16,
    num_epochs=3,
    warmup_ratio=0.1,
    weight_decay=0.01,
    evaluation_strategy="epoch"
)
```

**Implementation**:
```bash
python -m src.models.train_transformer --in data/processed/ --out experiments/runs/roberta_cross_encoder/
```

## Addressing Class Imbalance

### Techniques
1. **Class weights**: Adjust loss function to penalize minority class errors more heavily
2. **Focal loss**: Reduce weight of easy examples, focus on hard cases
3. **Probability calibration**: Post-hoc calibration using Platt scaling or isotonic regression

### Evaluation Focus
- Emphasize **Relevant class recall** (capture most relevant utterances)
- Use **AUPRC** instead of AUC-ROC for imbalanced data
- Monitor **per-class performance** not just overall accuracy

## Model Training Pipeline

### 1. Data Loading and Preprocessing
```python
# Load splits
train_df = pd.read_csv("data/processed/splits/train.csv")
val_df = pd.read_csv("data/processed/splits/val.csv")
test_df = pd.read_csv("data/processed/splits/test.csv")

# Prepare features
X_train = prepare_features(train_df)  # Motion + utterance text
y_train = train_df['label'].values
```

### 2. Model Training
- **Cross-validation**: Use validation set for hyperparameter tuning
- **Early stopping**: Prevent overfitting on transformer models
- **Checkpointing**: Save best models based on validation F1

### 3. Model Persistence
```python
# Save model artifacts
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")  # For baseline models

# Save metrics
with open("metrics.json", "w") as f:
    json.dump(evaluation_metrics, f)
```

## Hyperparameter Tuning

### Baseline Models
- **TF-IDF parameters**: max_features, ngram_range, min_df, max_df
- **Regularization**: C parameter for SVM/LogReg
- **Class weights**: Balance ratio between classes

### Transformer Models
- **Learning rate**: 1e-5, 2e-5, 5e-5
- **Batch size**: 8, 16, 32 (memory permitting)
- **Epochs**: 2, 3, 5 with early stopping
- **Max sequence length**: 256, 384, 512 tokens

## Model Explainability

### Baseline Models
- **Feature importance**: Top TF-IDF features for each class
- **Coefficient analysis**: Most predictive n-grams
- **Decision boundaries**: Visualize linear separability

### Transformer Models
- **Attention visualization**: Which tokens the model focuses on
- **Gradient-based saliency**: Important words for classification
- **LIME/SHAP**: Local explanations for individual predictions

## Evaluation Metrics

### Primary Metrics
- **Macro-F1**: Balanced performance across both classes
- **Relevant Recall**: Capture rate for relevant utterances
- **AUPRC**: Area under precision-recall curve (handles imbalance)

### Secondary Metrics
- **Balanced Accuracy**: Account for class imbalance
- **Precision-Recall curves**: Visualize trade-offs
- **Confusion matrices**: Error pattern analysis

### Slice Analysis
- **Per-sitting performance**: Generalization across sessions
- **Per-speaker performance**: Bias detection
- **Per-motion-type performance**: Domain adaptation

## Expected Outcomes

### Performance Targets
- **Baseline models**: Macro-F1 ≥ 0.60, Relevant Recall ≥ 0.70
- **Transformer models**: Macro-F1 ≥ 0.75, Relevant Recall ≥ 0.80
- **Improvement**: Transformer should outperform baseline by ≥5% F1

### Model Artifacts
```
experiments/runs/
├── baseline_svm/
│   ├── logistic_regression/
│   │   ├── model.pkl
│   │   ├── vectorizer.pkl
│   │   └── metrics.json
│   └── svm/
│       ├── model.pkl
│       ├── vectorizer.pkl
│       └── metrics.json
└── roberta_cross_encoder/
    ├── pytorch_model.bin
    ├── config.json
    ├── tokenizer_config.json
    └── metrics.json
```

## Risk Mitigation

### Overfitting Prevention
- **Validation monitoring**: Stop training when validation performance plateaus
- **Regularization**: L2 penalty for linear models, dropout for transformers
- **Cross-validation**: Ensure consistent performance across folds

### Computational Constraints
- **CPU fallback**: Ensure models work without GPU
- **Memory optimization**: Gradient checkpointing, smaller batch sizes
- **Time limits**: Set maximum training time per model

### Reproducibility
- **Random seeds**: Fixed seeds for all random operations
- **Environment**: Document exact package versions
- **Data versioning**: Hash training data to detect changes

## Success Criteria

This phase succeeds when:
1. **Baseline established**: Working TF-IDF + linear models with reasonable performance
2. **Transformer trained**: RoBERTa model outperforms baselines
3. **Metrics achieved**: Primary KPIs met on validation set
4. **Artifacts saved**: All models and metrics properly persisted
5. **Explainability**: Model decisions can be interpreted and explained

---

**Next Phase:** [EV] Evaluation - Comprehensive performance analysis and model selection
