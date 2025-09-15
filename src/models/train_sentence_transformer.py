"""
Train a sentence transformer-based classifier for parliamentary relevance classification.

Usage:
    python -m src.models.train_sentence_transformer --in data/processed/ --out experiments/runs/sentence_transformer/
"""

import typer
from pathlib import Path
import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
import joblib
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore")


app = typer.Typer()


@app.command()
def main(
    input_dir: Path = typer.Option(..., "--in", help="Input directory with processed data"),
    out: Path = typer.Option(..., help="Output directory for model artifacts"),
    model_name: str = typer.Option("all-MiniLM-L6-v2", help="Sentence transformer model name"),
    test_size: float = typer.Option(0.2, help="Test set proportion"),
    random_state: int = typer.Option(42, help="Random seed"),
    classifiers: str = typer.Option("logistic,svm,rf", help="Comma-separated list of classifiers to train"),
):
    """
    Train sentence transformer-based classifiers for parliamentary relevance classification.
    
    This uses a pre-trained sentence transformer to encode motion-utterance pairs,
    then trains traditional ML classifiers on the embeddings.
    """
    typer.echo(f"[MO] Training sentence transformer classifiers from {input_dir}")
    typer.echo(f"Model: {model_name}")
    typer.echo(f"Output directory: {out}")
    
    out.mkdir(parents=True, exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(random_state)
    
    # Load labeled data
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
    
    # Prepare text features
    X_text = prepare_text_features(df)
    y = df[label_column].values
    
    # Check class distribution
    unique_labels, counts = np.unique(y, return_counts=True)
    typer.echo(f"Class distribution: {dict(zip(unique_labels, counts))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    typer.echo(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Load sentence transformer
    typer.echo("Loading sentence transformer...")
    sentence_model = SentenceTransformer(model_name)
    
    # Generate embeddings
    typer.echo("Generating embeddings for training data...")
    X_train_embeddings = sentence_model.encode(X_train, show_progress_bar=True)
    
    typer.echo("Generating embeddings for test data...")
    X_test_embeddings = sentence_model.encode(X_test, show_progress_bar=True)
    
    typer.echo(f"Embedding dimensions: {X_train_embeddings.shape[1]}")
    
    # Parse classifier names
    classifier_names = [name.strip() for name in classifiers.split(',')]
    
    # Train classifiers
    results = {}
    
    for classifier_name in classifier_names:
        typer.echo(f"\nTraining {classifier_name} classifier...")
        
        # Initialize classifier
        if classifier_name == "logistic":
            clf = LogisticRegression(random_state=random_state, max_iter=1000)
        elif classifier_name == "svm":
            clf = SVC(random_state=random_state, probability=True)
        elif classifier_name == "rf":
            clf = RandomForestClassifier(random_state=random_state, n_estimators=100)
        else:
            typer.echo(f"Unknown classifier: {classifier_name}")
            continue
        
        # Train classifier
        clf.fit(X_train_embeddings, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test_embeddings)
        y_pred_proba = clf.predict_proba(X_test_embeddings) if hasattr(clf, 'predict_proba') else None
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True, target_names=['NOT_RELEVANT', 'RELEVANT'])
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate additional metrics
        accuracy = report['accuracy']
        macro_f1 = report['macro avg']['f1-score']
        relevant_recall = report.get('RELEVANT', {}).get('recall', 0)
        relevant_precision = report.get('RELEVANT', {}).get('precision', 0)
        not_relevant_recall = report.get('NOT_RELEVANT', {}).get('recall', 0)
        not_relevant_precision = report.get('NOT_RELEVANT', {}).get('precision', 0)
        
        # Save model and results
        model_dir = out / classifier_name
        model_dir.mkdir(exist_ok=True)
        
        # Save classifier
        joblib.dump(clf, model_dir / "classifier.pkl")
        
        # Save sentence transformer
        sentence_model.save(str(model_dir / "sentence_transformer"))
        
        # Save metrics
        metrics = {
            'model_name': f'sentence_transformer_{classifier_name}',
            'base_model': model_name,
            'test_accuracy': float(accuracy),
            'macro_f1': float(macro_f1),
            'relevant_recall': float(relevant_recall),
            'relevant_precision': float(relevant_precision),
            'not_relevant_recall': float(not_relevant_recall),
            'not_relevant_precision': float(not_relevant_precision),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'embedding_dimensions': int(X_train_embeddings.shape[1]),
            'label_distribution': {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
        }
        
        with open(model_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        results[classifier_name] = metrics
        
        typer.echo(f"✓ {classifier_name} classifier trained and saved to {model_dir}")
        typer.echo(f"✓ Test Accuracy: {accuracy:.3f}")
        typer.echo(f"✓ Macro F1: {macro_f1:.3f}")
        typer.echo(f"✓ RELEVANT Recall: {relevant_recall:.3f}")
        typer.echo(f"✓ NOT_RELEVANT Recall: {not_relevant_recall:.3f}")
    
    # Save combined results
    with open(out / "all_metrics.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    typer.echo(f"\n✓ All sentence transformer classifiers trained and saved to {out}")


def prepare_text_features(df: pd.DataFrame) -> List[str]:
    """Prepare text features by combining motion and utterance text."""
    features = []
    
    for _, row in df.iterrows():
        # Combine motion and utterance text
        motion_text = row.get('motion_text', '') or ''
        utterance_text = row.get('text', '') or ''
        
        # Create combined feature text with clear separation
        combined_text = f"Motion: {motion_text} [SEP] Utterance: {utterance_text}"
        features.append(combined_text)
    
    return features


if __name__ == "__main__":
    app()
