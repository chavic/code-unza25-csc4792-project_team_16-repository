"""
Create seed annotation set from utterances.

Usage:
    python -m src.label.make_seed --in data/interim/utterances.jsonl --n 1000 --out data/processed/seed.csv
"""

import typer
from pathlib import Path
import json
import pandas as pd
import random
from typing import List, Dict


app = typer.Typer()


@app.command()
def main(
    input_file: Path = typer.Option(..., "--in", help="Input JSONL file with utterances"),
    n: int = typer.Option(1000, help="Number of utterances to sample for annotation"),
    out: Path = typer.Option(..., help="Output CSV file for annotation"),
    seed: int = typer.Option(42, help="Random seed for reproducible sampling"),
):
    """
    Create a seed annotation set by sampling utterances for manual labeling.
    
    This creates a balanced sample of utterances that will be manually labeled
    as 'Relevant' or 'NotRelevant' to their associated motions.
    """
    typer.echo(f"[DP] Creating seed annotation set from {input_file}")
    typer.echo(f"Sampling {n} utterances for annotation")
    
    random.seed(seed)
    
    # Load utterances
    utterances = []
    with open(input_file) as f:
        for line in f:
            utterances.append(json.loads(line.strip()))
    
    typer.echo(f"Loaded {len(utterances)} total utterances")
    
    # Filter utterances that have linked motions
    linked_utterances = [u for u in utterances if u.get('motion_id')]
    typer.echo(f"Found {len(linked_utterances)} utterances with linked motions")
    
    # Sample utterances for annotation
    if len(linked_utterances) < n:
        typer.echo(f"Warning: Only {len(linked_utterances)} linked utterances available")
        sample_utterances = linked_utterances
    else:
        sample_utterances = random.sample(linked_utterances, n)
    
    # Create annotation DataFrame
    annotation_data = []
    for utterance in sample_utterances:
        annotation_data.append({
            'sitting_id': utterance['sitting_id'],
            'utterance_id': utterance['utterance_id'],
            'speaker': utterance['speaker'],
            'timestamp': utterance['timestamp'],
            'text': utterance['text'],
            'motion_id': utterance['motion_id'],
            'motion_text': utterance['motion_text'],
            'label': '',  # To be filled by annotators
            'annotator': '',  # To be filled by annotators
            'notes': '',  # Optional notes from annotators
            'confidence': '',  # Annotator confidence (1-5 scale)
        })
    
    # Save as CSV
    df = pd.DataFrame(annotation_data)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    
    typer.echo(f"✓ Created annotation set with {len(annotation_data)} utterances")
    typer.echo(f"Saved to: {out}")
    typer.echo("\nAnnotation Instructions:")
    typer.echo("- Label each utterance as 'Relevant' or 'NotRelevant'")
    typer.echo("- 'Relevant': argues for/against motion, provides evidence, proposes actions")
    typer.echo("- 'NotRelevant': procedural points, greetings, tangents, jokes")
    typer.echo("- Fill in annotator name and confidence (1-5)")
    typer.echo("- Add notes for unclear cases")


if __name__ == "__main__":
    app()
