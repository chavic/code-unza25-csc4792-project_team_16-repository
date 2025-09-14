"""
Train baseline models (SVM, Logistic Regression) with TF-IDF features.

Usage:
    python -m src.models.train_baselines --in data/processed/ --out experiments/runs/baseline_svm/
"""

import typer
from pathlib import Path
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import numpy as np


app = typer.Typer()


@app.command()
def main(
    input_dir: Path = typer.Option(..., "--in", help="Input directory with processed data"),
    out: Path = typer.Option(..., help="Output directory for model artifacts"),
    test_size: float = typer.Option(0.2, help="Test set proportion"),
    random_state: int = typer.Option(42, help="Random seed"),
):
    """
    Train baseline models using TF-IDF features.
    
    This trains both SVM and Logistic Regression classifiers on the labeled
    utterances using TF-IDF vectorization of the combined motion and utterance text.
    """
    typer.echo(f"[MO] Training baseline models from {input_dir}")
    typer.echo(f"Output directory: {out}")
    
    out.mkdir(parents=True, exist_ok=True)
    
    # Load labeled data - try multiple possible file names
    possible_files = [
        input_dir / "auto_annotated_large.csv",
        input_dir / "auto_annotated_improved.csv", 
        input_dir / "auto_annotated_test.csv",
        input_dir / "seed_annotation.csv",
        input_dir / "seed.csv"
    ]
    
    labeled_file = None
    for file_path in possible_files:
        if file_path.exists():
            labeled_file = file_path
            break
    
    if not labeled_file:
        typer.echo(f"Error: No labeled data file found. Tried: {[str(f) for f in possible_files]}")
        raise typer.Exit(1)
    
    typer.echo(f"Loading labeled data from: {labeled_file}")
    df = pd.read_csv(labeled_file)
    
    # Determine label column name
    label_column = None
    for col in ['llm_label', 'label']:
        if col in df.columns:
            label_column = col
            break
    
    if not label_column:
        typer.echo(f"Error: No label column found. Available columns: {list(df.columns)}")
        raise typer.Exit(1)
    
    # Filter out unlabeled rows
    df = df[df[label_column].notna() & (df[label_column] != '') & (df[label_column] != 'UNKNOWN')]
    typer.echo(f"Loaded {len(df)} labeled utterances using column '{label_column}'")
    
    # Prepare features and labels
    X_text = prepare_text_features(df)
    y = df[label_column].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    typer.echo(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Vectorize text
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.95
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train models
    models = {
        'logistic_regression': LogisticRegression(random_state=random_state, max_iter=1000),
        'svm': SVC(random_state=random_state, probability=True)
    }
    
    results = {}
    
    for model_name, model in models.items():
        typer.echo(f"Training {model_name}...")
        
        # Train model
        model.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_vec)
        y_pred_proba = model.predict_proba(X_test_vec)
        
        # Generate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # Save model artifacts
        model_dir = out / model_name
        model_dir.mkdir(exist_ok=True)
        
        joblib.dump(model, model_dir / "model.pkl")
        joblib.dump(vectorizer, model_dir / "vectorizer.pkl")
        
        # Save metrics
        label_dist = dict(zip(*np.unique(y, return_counts=True)))
        # Convert numpy types to Python types for JSON serialization
        label_dist = {k: int(v) for k, v in label_dist.items()}
        
        metrics = {
            'model_name': model_name,
            'test_accuracy': float(report['accuracy']),
            'macro_f1': float(report['macro avg']['f1-score']),
            'relevant_recall': float(report.get('RELEVANT', {}).get('recall', 0)),
            'relevant_precision': float(report.get('RELEVANT', {}).get('precision', 0)),
            'not_relevant_recall': float(report.get('NOT_RELEVANT', {}).get('recall', 0)),
            'not_relevant_precision': float(report.get('NOT_RELEVANT', {}).get('precision', 0)),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'feature_count': int(X_train_vec.shape[1]),
            'label_distribution': label_dist
        }
        
        with open(model_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        results[model_name] = metrics
        
        typer.echo(f"✓ {model_name} - Accuracy: {metrics['test_accuracy']:.3f}, Macro F1: {metrics['macro_f1']:.3f}")
    
    typer.echo(f"✓ Baseline models trained and saved to {out}")


def prepare_text_features(df: pd.DataFrame) -> list:
    """Prepare text features by combining motion and utterance text."""
    features = []
    
    for _, row in df.iterrows():
        # Combine motion and utterance text
        motion_text = row.get('motion_text', '') or ''
        utterance_text = row.get('text', '') or ''
        
        # Create combined feature text
        combined_text = f"{motion_text} [SEP] {utterance_text}"
        features.append(combined_text)
    
    return features


if __name__ == "__main__":
    app()
