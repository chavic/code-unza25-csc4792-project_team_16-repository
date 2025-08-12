"""
Generate evaluation reports and visualizations.

Usage:
    python -m src.eval.report --run experiments/runs/baseline_svm/ --out reports/figs/
"""

import typer
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


app = typer.Typer()


@app.command()
def main(
    run_dir: Path = typer.Option(..., help="Experiment run directory with metrics.json"),
    out: Path = typer.Option(..., help="Output directory for figures and reports"),
):
    """
    Generate evaluation reports and visualizations from experiment results.
    
    This creates confusion matrix plots, metrics tables, and error analysis
    visualizations from the trained model results.
    """
    typer.echo(f"[EV] Generating evaluation report from {run_dir}")
    typer.echo(f"Output directory: {out}")
    
    out.mkdir(parents=True, exist_ok=True)
    
    # Find all model results in the run directory
    model_results = []
    
    for model_dir in run_dir.iterdir():
        if model_dir.is_dir():
            metrics_file = model_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                    metrics['model_dir'] = str(model_dir)
                    model_results.append(metrics)
    
    if not model_results:
        typer.echo(f"No model results found in {run_dir}")
        raise typer.Exit(1)
    
    typer.echo(f"Found {len(model_results)} model results")
    
    # Generate comparison table
    generate_metrics_table(model_results, out)
    
    # Generate confusion matrices
    for result in model_results:
        generate_confusion_matrix(result, out)
    
    # Generate metrics comparison plot
    generate_metrics_comparison(model_results, out)
    
    typer.echo(f"✓ Evaluation reports generated in {out}")


def generate_metrics_table(results: list, out_dir: Path):
    """Generate a comparison table of model metrics."""
    table_data = []
    
    for result in results:
        table_data.append({
            'Model': result['model_name'],
            'Accuracy': f"{result['test_accuracy']:.3f}",
            'Macro F1': f"{result['macro_f1']:.3f}",
            'Relevant Recall': f"{result['relevant_recall']:.3f}",
            'Relevant Precision': f"{result['relevant_precision']:.3f}",
            'Features': result.get('feature_count', 'N/A')
        })
    
    df = pd.DataFrame(table_data)
    
    # Save as CSV
    df.to_csv(out_dir / "metrics_comparison.csv", index=False)
    
    # Create formatted table plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title("Model Performance Comparison", fontsize=14, fontweight='bold')
    plt.savefig(out_dir / "metrics_table.png", dpi=300, bbox_inches='tight')
    plt.close()


def generate_confusion_matrix(result: dict, out_dir: Path):
    """Generate confusion matrix visualization for a model."""
    cm = np.array(result['confusion_matrix'])
    model_name = result['model_name']
    
    # Create confusion matrix plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get class labels from the classification report
    labels = list(result['classification_report'].keys())
    labels = [l for l in labels if l not in ['accuracy', 'macro avg', 'weighted avg']]
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    
    ax.set_title(f'Confusion Matrix - {model_name.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(out_dir / f"confusion_matrix_{model_name}.png", dpi=300, bbox_inches='tight')
    plt.close()


def generate_metrics_comparison(results: list, out_dir: Path):
    """Generate bar plot comparing key metrics across models."""
    metrics = ['test_accuracy', 'macro_f1', 'relevant_recall', 'relevant_precision']
    metric_labels = ['Accuracy', 'Macro F1', 'Relevant Recall', 'Relevant Precision']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        
        model_names = [r['model_name'].replace('_', ' ').title() for r in results]
        values = [r[metric] for r in results]
        
        bars = ax.bar(model_names, values, color=['skyblue', 'lightcoral'])
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / "metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    app()
