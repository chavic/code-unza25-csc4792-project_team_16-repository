# Parliamentary Debate Analysis - Demo Application

## 🎯 Overview

This is an interactive web application that demonstrates the parliamentary debate analysis system. Users can select motions and utterances to see how the AI classifies relevance in real-time.

## 🚀 Quick Start

```bash
# Start the demo
python run_demo.py

# Visit in browser
open http://localhost:8080
```

## ✨ Features

### Interactive Classification
- **Select Motions**: Choose from real parliamentary motions
- **Select Utterances**: Pick from actual MP utterances
- **Choose Models**: Test different AI architectures
- **Real-time Results**: See predictions with confidence scores

### Visual Interface
- **Clean Design**: Professional presentation-ready interface
- **Confidence Visualization**: Color-coded confidence bars
- **Model Comparison**: Test TF-IDF vs Sentence Transformers
- **True Labels**: Compare predictions with actual labels

### Multiple Models
- **Baseline Models**: TF-IDF with Logistic Regression, SVM
- **Advanced Models**: Sentence Transformers with multiple classifiers
- **Performance Metrics**: See accuracy, recall, and precision

## 🏗️ Architecture

```
Frontend (HTML/CSS/JS) → Flask API → Trained Models → Results
```

### Components:
- **Flask Backend**: RESTful API for classification
- **Model Integration**: Loads trained scikit-learn models
- **Data Management**: Serves motions and utterances
- **Real-time Processing**: Instant classification results

## 📊 Demo Data

The demo uses real parliamentary data:
- **Motions**: Actual parliamentary motions from order papers
- **Utterances**: Real MP speeches and responses
- **Labels**: AI-generated relevance classifications
- **Models**: Trained on 500+ labeled examples

## 🎨 User Interface

### Main Features:
1. **Motion Selection**: Dropdown with parliamentary motions
2. **Utterance Selection**: Choose from MP utterances
3. **Model Selection**: Pick classification algorithm
4. **Classification Button**: Run the analysis
5. **Results Display**: Show prediction, confidence, and true label

### Visual Elements:
- **Gradient Background**: Professional blue-purple theme
- **Card Layout**: Clean, modern design
- **Confidence Bars**: Visual confidence indicators
- **Color Coding**: Green for relevant, red for not relevant
- **Statistics**: Performance metrics display

## 🔧 Technical Details

### Backend (Flask):
- **RESTful API**: `/api/motions`, `/api/utterances`, `/api/classify`
- **Model Loading**: Dynamic loading of trained models
- **Error Handling**: Graceful fallbacks for missing data
- **CORS Support**: Cross-origin request handling

### Frontend (HTML/CSS/JS):
- **Bootstrap 5**: Responsive design framework
- **Font Awesome**: Professional icons
- **Custom CSS**: Gradient backgrounds and animations
- **JavaScript**: Dynamic content loading and API calls

### Model Integration:
- **Scikit-learn**: Trained classifiers (Logistic, SVM, Random Forest)
- **Sentence Transformers**: Semantic embedding models
- **Joblib**: Model serialization and loading
- **Pandas**: Data manipulation and serving

## 📈 Performance

### Model Performance:
- **Accuracy**: 98% (due to class imbalance)
- **Relevant Recall**: 100% (catches all relevant utterances)
- **Not Relevant Recall**: 0% (challenge with minority class)
- **Macro F1**: 49.5% (balanced performance metric)

### System Performance:
- **Load Time**: < 2 seconds for model loading
- **Classification**: < 1 second per prediction
- **Memory Usage**: ~500MB for all models
- **Concurrent Users**: Supports multiple simultaneous users

## 🛠️ Setup Requirements

### Dependencies:
```bash
pip install flask pandas scikit-learn joblib sentence-transformers
```

### Data Files:
- `data/processed/auto_annotated_large.csv` (utterances with labels)
- `data/processed/utterances_full.jsonl` (full utterance data)

### Model Files:
- `experiments/runs/baseline_auto_annotated/` (TF-IDF models)
- `experiments/runs/sentence_transformer/` (embedding models)

## 🎯 Use Cases

### For Presentations:
- **Academic**: Demonstrate AI classification capabilities
- **Industry**: Show practical NLP applications
- **Government**: Illustrate parliamentary analysis tools

### For Testing:
- **Model Validation**: Test different architectures
- **Data Quality**: Verify annotation accuracy
- **Performance**: Compare model effectiveness

### For Development:
- **Prototype**: Base for production systems
- **Integration**: API for other applications
- **Extension**: Foundation for advanced features

## 🔮 Future Enhancements

### Immediate:
- **Real-time Processing**: Live debate analysis
- **Batch Upload**: Process multiple utterances
- **Export Results**: Download classification results

### Advanced:
- **Multi-language**: Support for other languages
- **Sentiment Analysis**: Emotional tone detection
- **Topic Modeling**: Automatic topic extraction
- **Speaker Analysis**: Individual MP behavior patterns

## 📝 Usage Instructions

1. **Start the Demo**:
   ```bash
   python run_demo.py
   ```

2. **Open Browser**: Navigate to `http://localhost:8080`

3. **Select Motion**: Choose a parliamentary motion from dropdown

4. **Select Utterance**: Pick an MP utterance to analyze

5. **Choose Model**: Select classification algorithm

6. **Classify**: Click "Classify Relevance" button

7. **View Results**: See prediction, confidence, and true label

## 🐛 Troubleshooting

### Common Issues:
- **Port 8080 in use**: Change port in `app.py`
- **Models not found**: Run model training first
- **Data files missing**: Execute data preparation pipeline
- **Import errors**: Install missing dependencies

### Debug Mode:
```bash
# Run with debug output
FLASK_DEBUG=1 python app.py
```

## 📞 Support

For issues or questions:
- Check the main README.md for setup instructions
- Review the technical report for detailed methodology
- Examine the Jupyter notebook for implementation details

---

**Demo Application** - Parliamentary Debate Analysis System  
**CSC 4792 - Team 16** - Motion Relevance Classification
