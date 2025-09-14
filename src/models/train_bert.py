"""
Train BERT-based classifier for parliamentary relevance classification.

Usage:
    python -m src.models.train_bert --in data/processed/ --out experiments/runs/bert_classifier/
"""

import typer
from pathlib import Path
import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModel, 
    TrainingArguments, Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset as HFDataset
import joblib
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore")


app = typer.Typer()


class ParliamentaryDataset(Dataset):
    """Custom dataset for parliamentary utterances."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label mapping
        self.label_map = {'NOT_RELEVANT': 0, 'RELEVANT': 1}
        self.reverse_label_map = {0: 'NOT_RELEVANT', 1: 'RELEVANT'}
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.label_map[self.labels[idx]]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ParliamentaryBERTClassifier(nn.Module):
    """BERT-based classifier for parliamentary relevance."""
    
    def __init__(self, model_name='distilbert-base-uncased', num_classes=2, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)


@app.command()
def main(
    input_dir: Path = typer.Option(..., "--in", help="Input directory with processed data"),
    out: Path = typer.Option(..., help="Output directory for model artifacts"),
    model_name: str = typer.Option("distilbert-base-uncased", help="HuggingFace model name"),
    test_size: float = typer.Option(0.2, help="Test set proportion"),
    random_state: int = typer.Option(42, help="Random seed"),
    max_length: int = typer.Option(512, help="Maximum sequence length"),
    batch_size: int = typer.Option(8, help="Batch size for training"),
    learning_rate: float = typer.Option(2e-5, help="Learning rate"),
    num_epochs: int = typer.Option(3, help="Number of training epochs"),
    dropout: float = typer.Option(0.3, help="Dropout rate"),
):
    """
    Train a BERT-based classifier for parliamentary relevance classification.
    
    This uses a pre-trained transformer model (DistilBERT by default) to classify
    utterances as relevant or not relevant to their associated motions.
    """
    typer.echo(f"[MO] Training BERT classifier from {input_dir}")
    typer.echo(f"Model: {model_name}")
    typer.echo(f"Output directory: {out}")
    
    out.mkdir(parents=True, exist_ok=True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(random_state)
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
    
    # Initialize tokenizer
    typer.echo("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    train_dataset = ParliamentaryDataset(X_train, y_train, tokenizer, max_length)
    test_dataset = ParliamentaryDataset(X_test, y_test, tokenizer, max_length)
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    typer.echo(f"Class weights: {class_weight_dict}")
    
    # Initialize model
    typer.echo("Initializing BERT model...")
    model = ParliamentaryBERTClassifier(model_name, num_classes=2, dropout=dropout)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(out / "training_output"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=learning_rate,
        logging_dir=str(out / "logs"),
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        seed=random_state,
    )
    
    # Custom compute metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train model
    typer.echo("Starting training...")
    trainer.train()
    
    # Evaluate on test set
    typer.echo("Evaluating on test set...")
    eval_results = trainer.evaluate()
    
    # Get predictions for detailed metrics
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    
    # Generate detailed classification report
    report = classification_report(y_true, y_pred, output_dict=True, target_names=['NOT_RELEVANT', 'RELEVANT'])
    cm = confusion_matrix(y_true, y_pred)
    
    # Save model and tokenizer
    model_dir = out / "model"
    model_dir.mkdir(exist_ok=True)
    
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    
    # Save metrics
    metrics = {
        'model_name': 'bert_classifier',
        'base_model': model_name,
        'test_accuracy': float(eval_results['eval_accuracy']),
        'macro_f1': float(eval_results['eval_f1']),
        'macro_precision': float(eval_results['eval_precision']),
        'macro_recall': float(eval_results['eval_recall']),
        'relevant_recall': float(report.get('RELEVANT', {}).get('recall', 0)),
        'relevant_precision': float(report.get('RELEVANT', {}).get('precision', 0)),
        'not_relevant_recall': float(report.get('NOT_RELEVANT', {}).get('recall', 0)),
        'not_relevant_precision': float(report.get('NOT_RELEVANT', {}).get('precision', 0)),
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'class_weights': class_weight_dict,
        'training_args': {
            'max_length': max_length,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'dropout': dropout
        },
        'label_distribution': dict(zip(*np.unique(y, return_counts=True)))
    }
    
    with open(model_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    typer.echo(f"✓ BERT classifier trained and saved to {out}")
    typer.echo(f"✓ Test Accuracy: {metrics['test_accuracy']:.3f}")
    typer.echo(f"✓ Macro F1: {metrics['macro_f1']:.3f}")
    typer.echo(f"✓ RELEVANT Recall: {metrics['relevant_recall']:.3f}")
    typer.echo(f"✓ NOT_RELEVANT Recall: {metrics['not_relevant_recall']:.3f}")


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
