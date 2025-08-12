# Deployment (DE)

**Phase:** [DE] Deployment  
**Date:** 2024  
**Team:** Team 16  

## Overview

This phase creates production-ready tools for parliamentary relevance classification, including a command-line interface and interactive demonstration system.

## Deployment Architecture

### Core Components

1. **CLI Tool** - Batch processing for production use
2. **Colab Demo** - Interactive demonstration and validation
3. **Model Serving** - Efficient inference pipeline
4. **Documentation** - User guides and API documentation

### System Requirements
- **Python**: 3.10-3.11
- **Memory**: 4GB RAM minimum (8GB recommended for transformer models)
- **Storage**: 2GB for models and dependencies
- **CPU**: Multi-core recommended for batch processing

## Command-Line Interface (CLI)

### Core Functionality
```bash
python -m src.app.cli \
  --motion data/interim/order_paper_2023_03.txt \
  --transcript data/raw/sitting_2023_001.html \
  --out experiments/runs/production/labels.csv \
  --model-dir experiments/runs/roberta_cross_encoder/ \
  --threshold 0.6
```

### CLI Features

**Input Flexibility**:
- Motion text from file or Order Paper JSON
- Transcript from HTML or structured formats
- Batch processing of multiple files

**Output Options**:
- CSV with predictions and confidence scores
- JSON format for API integration
- Summary statistics and classification distribution

**Model Configuration**:
- Model selection (baseline vs transformer)
- Custom classification thresholds
- Confidence score calibration

### CLI Implementation
```python
@app.command()
def classify(
    motion: Path,
    transcript: Path, 
    out: Path,
    model_dir: Path = "experiments/runs/best_model/",
    threshold: float = 0.5,
    format: str = "csv",
    batch_size: int = 32
):
    """Classify utterances as relevant/not relevant to motion."""
```

**Error Handling**:
- Graceful failure for malformed inputs
- Detailed error messages with suggestions
- Validation of input file formats
- Model loading verification

**Performance Optimization**:
- Batch processing for multiple utterances
- Memory-efficient text processing
- Progress bars for long-running tasks
- Parallel processing where applicable

## Interactive Demonstration

### Google Colab Notebook

**Structure**:
```
notebooks/code-unza25-csc4792-project_team_N-notebook.ipynb

1. Introduction and Problem Statement
2. Data Loading and Exploration  
3. Model Training Pipeline
4. Evaluation Results
5. Interactive Demo
6. Conclusions and Future Work
```

**Interactive Features**:
- **Paste & Predict**: Input motion and utterance text for instant classification
- **Confidence Visualization**: Show prediction probabilities
- **Explanation**: Highlight important words/phrases
- **Batch Demo**: Process sample parliamentary excerpts

### Demo Implementation
```python
def interactive_demo():
    """Interactive widget for testing classifications."""
    motion_input = widgets.Textarea(
        placeholder="Enter motion text...",
        description="Motion:"
    )
    
    utterance_input = widgets.Textarea(
        placeholder="Enter speaker utterance...",
        description="Utterance:"
    )
    
    classify_button = widgets.Button(
        description="Classify",
        button_style="primary"
    )
    
    output = widgets.Output()
    
    def on_classify_click(b):
        with output:
            result = classify_utterance(
                motion_input.value,
                utterance_input.value
            )
            display_prediction(result)
    
    classify_button.on_click(on_classify_click)
```

**Demo Examples**:
- **Clear Relevant**: Direct support/opposition to motion
- **Clear NotRelevant**: Procedural or off-topic statements  
- **Ambiguous Cases**: Borderline examples with explanations
- **Error Cases**: Common failure modes with analysis

## Model Serving Infrastructure

### Model Loading and Caching
```python
class RelevanceClassifier:
    def __init__(self, model_dir: Path):
        self.model = self.load_model(model_dir)
        self.vectorizer = self.load_vectorizer(model_dir)
        self.threshold = self.load_threshold(model_dir)
    
    @lru_cache(maxsize=1000)
    def predict(self, motion_text: str, utterance_text: str) -> dict:
        """Cached prediction for efficiency."""
        features = self.prepare_features(motion_text, utterance_text)
        probabilities = self.model.predict_proba(features)[0]
        
        return {
            'label': 'Relevant' if probabilities[1] > self.threshold else 'NotRelevant',
            'confidence': float(max(probabilities)),
            'relevant_probability': float(probabilities[1]),
            'not_relevant_probability': float(probabilities[0])
        }
```

### Batch Processing Pipeline
```python
def process_transcript_batch(
    transcript_file: Path,
    motion_text: str,
    classifier: RelevanceClassifier,
    batch_size: int = 32
) -> List[dict]:
    """Efficiently process entire transcript."""
    
    utterances = parse_transcript(transcript_file)
    results = []
    
    for batch in chunked(utterances, batch_size):
        batch_predictions = classifier.predict_batch([
            (motion_text, utt['text']) for utt in batch
        ])
        
        for utterance, prediction in zip(batch, batch_predictions):
            results.append({
                **utterance,
                **prediction,
                'motion_text': motion_text
            })
    
    return results
```

## Output Formats and Integration

### CSV Output Format
```csv
sitting_id,speaker,timestamp,text,label,confidence,relevant_probability,motion_id
sitting_2023_001,Hon. Member A,2023-03-15T14:30:00,"I support this motion...",Relevant,0.89,0.89,M2023_001
sitting_2023_001,Hon. Member B,2023-03-15T14:31:00,"Point of order!",NotRelevant,0.92,0.08,M2023_001
```

### JSON API Format
```json
{
  "predictions": [
    {
      "utterance_id": "sitting_2023_001_042",
      "speaker": "Hon. Member A",
      "text": "I support this motion...",
      "label": "Relevant",
      "confidence": 0.89,
      "probabilities": {
        "relevant": 0.89,
        "not_relevant": 0.11
      }
    }
  ],
  "summary": {
    "total_utterances": 156,
    "relevant_count": 89,
    "not_relevant_count": 67,
    "average_confidence": 0.84
  },
  "metadata": {
    "model_version": "roberta_cross_encoder_v1",
    "threshold": 0.6,
    "processing_time": 12.3
  }
}
```

## Performance and Scalability

### Optimization Strategies
- **Model quantization**: Reduce memory footprint for deployment
- **Batch inference**: Process multiple utterances simultaneously
- **Caching**: Store results for repeated queries
- **Parallel processing**: Utilize multiple CPU cores

### Benchmarking
```python
def benchmark_inference():
    """Measure inference performance."""
    test_cases = load_test_utterances(100)
    
    start_time = time.time()
    predictions = classifier.predict_batch(test_cases)
    end_time = time.time()
    
    throughput = len(test_cases) / (end_time - start_time)
    print(f"Throughput: {throughput:.1f} utterances/second")
```

**Performance Targets**:
- **Baseline models**: >100 utterances/second
- **Transformer models**: >10 utterances/second
- **Memory usage**: <2GB for single session
- **Startup time**: <30 seconds for model loading

## Quality Assurance and Monitoring

### Input Validation
```python
def validate_inputs(motion_text: str, utterance_text: str) -> None:
    """Validate input format and content."""
    if not motion_text.strip():
        raise ValueError("Motion text cannot be empty")
    
    if not utterance_text.strip():
        raise ValueError("Utterance text cannot be empty")
    
    if len(motion_text) > 10000:
        warnings.warn("Motion text is very long, may affect performance")
    
    if len(utterance_text) > 5000:
        warnings.warn("Utterance text is very long, may be truncated")
```

### Prediction Confidence Monitoring
- **Low confidence alerts**: Flag predictions with confidence < 0.6
- **Distribution monitoring**: Track prediction distribution over time
- **Error logging**: Capture and analyze processing failures

### Model Version Management
```python
class ModelRegistry:
    """Manage multiple model versions."""
    
    def __init__(self):
        self.models = {}
        self.default_model = None
    
    def register_model(self, name: str, model_path: Path, is_default: bool = False):
        """Register a new model version."""
        self.models[name] = self.load_model(model_path)
        if is_default:
            self.default_model = name
    
    def predict(self, model_name: str = None, **kwargs):
        """Predict using specified or default model."""
        model_name = model_name or self.default_model
        return self.models[model_name].predict(**kwargs)
```

## Documentation and User Guides

### CLI Documentation
```markdown
# Parliament Relevance Classifier CLI

## Installation
pip install -r requirements.txt

## Basic Usage
python -m src.app.cli --motion motion.txt --transcript sitting.html --out results.csv

## Advanced Options
--threshold 0.7          # Custom classification threshold
--model-dir path/to/model # Use specific model
--format json            # Output format (csv/json)
--batch-size 64          # Batch processing size
```

### API Documentation
- **Function signatures**: Complete parameter documentation
- **Usage examples**: Common use cases with code snippets
- **Error handling**: Expected exceptions and solutions
- **Performance notes**: Optimization recommendations

## Deployment Checklist

### Pre-Deployment Testing
- [ ] CLI works with sample data
- [ ] Colab notebook runs end-to-end
- [ ] Model loading and inference functional
- [ ] Error handling covers edge cases
- [ ] Performance meets targets

### Production Readiness
- [ ] Input validation comprehensive
- [ ] Output format consistent
- [ ] Documentation complete
- [ ] Version control implemented
- [ ] Monitoring and logging active

### User Acceptance
- [ ] Demo successfully shows capabilities
- [ ] CLI interface intuitive for target users
- [ ] Output format meets requirements
- [ ] Performance acceptable for use cases

## Deliverables

### Software Components
1. **CLI Tool**: `src/app/cli.py` with full functionality
2. **Colab Notebook**: Interactive demonstration and documentation
3. **Model Serving**: Efficient inference pipeline
4. **Documentation**: User guides and API reference

### Demonstration Materials
1. **Live Demo**: Working Colab notebook with examples
2. **CLI Examples**: Command-line usage demonstrations
3. **Performance Metrics**: Speed and accuracy benchmarks
4. **Use Case Scenarios**: Real-world application examples

### Deployment Package
```
deployment/
├── models/           # Trained model artifacts
├── cli/             # Command-line interface
├── docs/            # User documentation
├── examples/        # Sample data and usage
└── requirements.txt # Dependencies
```

## Success Criteria

This phase succeeds when:
1. **Working CLI**: Fully functional command-line tool
2. **Interactive Demo**: Colab notebook demonstrates capabilities
3. **Production Ready**: Handles real-world inputs reliably
4. **Performance Targets**: Meets speed and accuracy requirements
5. **User Documentation**: Complete guides for all features
6. **Demonstration Ready**: Prepared for final presentation

---

**Project Complete**: All CRISP-DM phases delivered with working system
