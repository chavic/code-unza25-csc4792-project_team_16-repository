"""
Command-line interface for relevance classification.

Usage:
    python -m src.app.cli --motion data/interim/motion.txt --transcript data/raw/sitting.html --out labels.csv
"""

import typer
from pathlib import Path
import pandas as pd
import json
from bs4 import BeautifulSoup
import joblib
from typing import List, Dict, Any


app = typer.Typer()


@app.command()
def main(
    motion: Path = typer.Option(..., help="Path to motion text file"),
    transcript: Path = typer.Option(..., help="Path to transcript HTML file"),
    out: Path = typer.Option(..., help="Output CSV file for predictions"),
    model_dir: Path = typer.Option("experiments/runs/baseline_svm/logistic_regression", 
                                  help="Directory containing trained model"),
    threshold: float = typer.Option(0.5, help="Classification threshold"),
):
    """
    Classify utterances in a parliamentary transcript as relevant/not relevant to a motion.
    
    This is the main deployment interface for the relevance classifier.
    It processes a single transcript and motion pair to produce predictions.
    """
    typer.echo(f"[DE] Classifying relevance for motion and transcript")
    typer.echo(f"Motion: {motion}")
    typer.echo(f"Transcript: {transcript}")
    typer.echo(f"Model: {model_dir}")
    
    # Load motion text
    motion_text = load_motion_text(motion)
    typer.echo(f"Motion loaded: {motion_text[:100]}...")
    
    # Parse transcript
    utterances = parse_transcript(transcript)
    typer.echo(f"Found {len(utterances)} utterances in transcript")
    
    # Load model
    model, vectorizer = load_model(model_dir)
    typer.echo("Model loaded successfully")
    
    # Make predictions
    predictions = classify_utterances(utterances, motion_text, model, vectorizer, threshold)
    
    # Save results
    results_df = pd.DataFrame(predictions)
    out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out, index=False)
    
    # Print summary
    relevant_count = sum(1 for p in predictions if p['label'] == 'Relevant')
    typer.echo(f"✓ Classification complete")
    typer.echo(f"Results saved to: {out}")
    typer.echo(f"Summary: {relevant_count}/{len(predictions)} utterances classified as Relevant")


def load_motion_text(motion_file: Path) -> str:
    """Load motion text from file."""
    if motion_file.suffix == '.json':
        with open(motion_file) as f:
            data = json.load(f)
            # Assume first motion if multiple
            return data['motions'][0]['text']
    else:
        # Plain text file
        return motion_file.read_text().strip()


def parse_transcript(transcript_file: Path) -> List[Dict[str, Any]]:
    """Parse transcript HTML and extract utterances."""
    with open(transcript_file) as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    
    utterances = []
    speaker_divs = soup.find_all('div', class_='speaker')
    
    for i, speaker_div in enumerate(speaker_divs):
        text = speaker_div.get_text().strip()
        
        if ':' in text:
            speaker, utterance_text = text.split(':', 1)
            speaker = speaker.strip()
            utterance_text = utterance_text.strip()
        else:
            speaker = "Unknown"
            utterance_text = text
        
        utterances.append({
            'utterance_id': f"utt_{i:03d}",
            'speaker': speaker,
            'text': utterance_text,
            'timestamp': f"2023-01-01T{10 + i//10:02d}:{(i*2) % 60:02d}:00"
        })
    
    return utterances


def load_model(model_dir: Path):
    """Load trained model and vectorizer."""
    model = joblib.load(model_dir / "model.pkl")
    vectorizer = joblib.load(model_dir / "vectorizer.pkl")
    return model, vectorizer


def classify_utterances(utterances: List[Dict], motion_text: str, model, vectorizer, threshold: float) -> List[Dict]:
    """Classify utterances as relevant/not relevant."""
    predictions = []
    
    for utterance in utterances:
        # Prepare feature text
        combined_text = f"{motion_text} [SEP] {utterance['text']}"
        
        # Vectorize
        X = vectorizer.transform([combined_text])
        
        # Predict
        proba = model.predict_proba(X)[0]
        confidence = max(proba)
        
        # Assume binary classification with classes ['NotRelevant', 'Relevant']
        relevant_proba = proba[1] if len(proba) > 1 else proba[0]
        label = 'Relevant' if relevant_proba >= threshold else 'NotRelevant'
        
        predictions.append({
            'sitting_id': transcript_file.stem if 'transcript_file' in locals() else 'unknown',
            'speaker': utterance['speaker'],
            'timestamp': utterance['timestamp'],
            'text': utterance['text'],
            'label': label,
            'confidence': float(confidence)
        })
    
    return predictions


@app.command()
def demo():
    """
    Interactive demo for testing the classifier.
    """
    typer.echo("Parliament Relevance Classifier - Interactive Demo")
    typer.echo("=" * 50)
    
    # Sample motion and utterances for demo
    sample_motion = "That this House calls upon the Government to implement comprehensive economic reforms to address unemployment and promote sustainable development."
    
    sample_utterances = [
        "Mr. Speaker, I rise to support this motion as unemployment has reached critical levels in our constituencies.",
        "Thank you, Mr. Speaker. I would like to thank the honorable member for bringing this important motion.",
        "The economic data clearly shows that we need urgent reforms to create jobs for our youth.",
        "Point of order, Mr. Speaker! The honorable member is not addressing the motion.",
        "I propose we establish a special committee to oversee the implementation of these economic reforms."
    ]
    
    typer.echo(f"Sample Motion: {sample_motion}")
    typer.echo("\nSample Utterances:")
    
    for i, utterance in enumerate(sample_utterances, 1):
        typer.echo(f"{i}. {utterance}")
    
    typer.echo("\n" + "=" * 50)
    typer.echo("In a real deployment, these would be classified using the trained model.")
    typer.echo("Expected classifications:")
    typer.echo("1. Relevant (supports motion)")
    typer.echo("2. NotRelevant (procedural)")
    typer.echo("3. Relevant (provides evidence)")
    typer.echo("4. NotRelevant (procedural point)")
    typer.echo("5. Relevant (proposes action)")


if __name__ == "__main__":
    app()
