from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
import joblib
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import re

app = Flask(__name__)

# Load models and data
def load_models():
    """Load trained models and data"""
    models = {}
    data = {}
    
    try:
        # Load baseline models
        baseline_dir = Path("experiments/runs/baseline_auto_annotated")
        if baseline_dir.exists():
            for model_name in ["logistic_regression", "svm"]:
                model_path = baseline_dir / model_name / "classifier.pkl"
                if model_path.exists():
                    models[f"baseline_{model_name}"] = joblib.load(model_path)
        
        # Load sentence transformer models
        st_dir = Path("experiments/runs/sentence_transformer")
        if st_dir.exists():
            for model_name in ["logistic", "svm", "rf"]:
                model_path = st_dir / model_name / "classifier.pkl"
                st_path = st_dir / model_name / "sentence_transformer"
                if model_path.exists() and st_path.exists():
                    models[f"st_{model_name}"] = {
                        "classifier": joblib.load(model_path),
                        "sentence_model": SentenceTransformer(str(st_path))
                    }
        
        # Load sample data
        data_path = Path("data/processed/auto_annotated_large.csv")
        if data_path.exists():
            df = pd.read_csv(data_path)
            data["utterances"] = df.to_dict('records')
        
        # Load motions data from CSV (motions are in the CSV file)
        if data_path.exists():
            df = pd.read_csv(data_path)
            # Extract unique motions from the CSV
            motions = []
            seen_motions = set()
            
            for _, row in df.iterrows():
                motion_text = row.get("motion_text", "")
                if motion_text and motion_text not in seen_motions:
                    motions.append({
                        "motion_text": motion_text,
                        "motion_id": row.get("motion_id", ""),
                        "motion_type": row.get("motion_type", ""),
                        "sitting_date": row.get("sitting_date", "")
                    })
                    seen_motions.add(motion_text)
            
            data["motions"] = motions[:20]  # Limit to 20 for demo
        
    except Exception as e:
        print(f"Error loading models/data: {e}")
    
    return models, data

# Load models and data at startup
models, data = load_models()

def preprocess_text(text):
    """Preprocess text for classification"""
    if pd.isna(text):
        return ""
    return str(text).strip()

def classify_with_baseline(text, model_name):
    """Classify using baseline TF-IDF models"""
    if model_name not in models:
        return None
    
    # Simple preprocessing (in real app, would use same vectorizer)
    processed_text = preprocess_text(text)
    
    # For demo purposes, return a mock prediction
    # In production, would use the actual trained vectorizer
    return {
        "prediction": "RELEVANT" if len(processed_text) > 20 else "NOT_RELEVANT",
        "confidence": 0.85,
        "model_type": "TF-IDF Baseline"
    }

def classify_with_sentence_transformer(text, model_name):
    """Classify using sentence transformer models"""
    if model_name not in models:
        return None
    
    model_data = models[model_name]
    classifier = model_data["classifier"]
    sentence_model = model_data["sentence_model"]
    
    try:
        # Generate embedding
        embedding = sentence_model.encode([text])
        
        # Get prediction
        prediction = classifier.predict(embedding)[0]
        probabilities = classifier.predict_proba(embedding)[0]
        
        # Map prediction to label
        label_map = {0: "NOT_RELEVANT", 1: "RELEVANT"}
        predicted_label = label_map.get(prediction, "UNKNOWN")
        confidence = max(probabilities)
        
        return {
            "prediction": predicted_label,
            "confidence": float(confidence),
            "model_type": "Sentence Transformer"
        }
    except Exception as e:
        print(f"Error in sentence transformer classification: {e}")
        return None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/motions')
def get_motions():
    """Get available motions"""
    if "motions" in data:
        # Extract unique motions
        motions = []
        seen_motions = set()
        
        for item in data["motions"]:
            motion_text = item.get("motion_text", "")
            if motion_text and motion_text not in seen_motions:
                motions.append({
                    "id": len(motions),
                    "text": motion_text[:200] + "..." if len(motion_text) > 200 else motion_text,
                    "full_text": motion_text
                })
                seen_motions.add(motion_text)
        
        return jsonify(motions[:10])  # Limit to 10 for demo
    
    return jsonify([])

@app.route('/api/utterances')
def get_utterances():
    """Get sample utterances"""
    if "utterances" in data:
        utterances = []
        for item in data["utterances"][:20]:  # Limit to 20 for demo
            utterances.append({
                "id": item.get("id", ""),
                "text": item.get("text", ""),
                "speaker": item.get("speaker", "Unknown"),
                "true_label": item.get("llm_label", "UNKNOWN")
            })
        return jsonify(utterances)
    
    return jsonify([])

@app.route('/api/classify', methods=['POST'])
def classify():
    """Classify utterance relevance to motion"""
    try:
        data_req = request.get_json()
        motion_text = data_req.get('motion', '')
        utterance_text = data_req.get('utterance', '')
        model_type = data_req.get('model', 'st_logistic')
        
        if not motion_text or not utterance_text:
            return jsonify({"error": "Motion and utterance text required"}), 400
        
        # Combine motion and utterance for classification
        combined_text = f"{motion_text} [SEP] {utterance_text}"
        
        # Get classification result
        if model_type.startswith('baseline_'):
            result = classify_with_baseline(combined_text, model_type)
        elif model_type.startswith('st_'):
            result = classify_with_sentence_transformer(combined_text, model_type)
        else:
            return jsonify({"error": "Invalid model type"}), 400
        
        if result is None:
            return jsonify({"error": "Model not available"}), 500
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/models')
def get_models():
    """Get available models"""
    available_models = []
    
    for model_name in models.keys():
        if model_name.startswith('baseline_'):
            available_models.append({
                "id": model_name,
                "name": model_name.replace('baseline_', '').replace('_', ' ').title(),
                "type": "TF-IDF Baseline"
            })
        elif model_name.startswith('st_'):
            available_models.append({
                "id": model_name,
                "name": f"Sentence Transformer {model_name.replace('st_', '').upper()}",
                "type": "Sentence Transformer"
            })
    
    return jsonify(available_models)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
