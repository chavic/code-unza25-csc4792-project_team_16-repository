"""
Segment transcripts into utterances and link to motions.

Usage:
    python -m src.parse.segment --in data/raw/ --order-papers data/interim/ --out data/interim/utterances.jsonl
"""

import typer
from pathlib import Path
import json
from bs4 import BeautifulSoup
from tqdm import tqdm
import re
from typing import List, Dict, Any


app = typer.Typer()


@app.command()
def main(
    input_dir: Path = typer.Option(..., "--in", help="Input directory with raw sitting files"),
    order_papers_dir: Path = typer.Option(..., help="Directory with order papers"),
    out: Path = typer.Option(..., help="Output JSONL file for utterances"),
):
    """
    Segment parliamentary transcripts into speaker utterances and link to motions.
    
    This processes the raw HTML files from scraping and creates structured
    utterances with speaker, timestamp, text, and linked motion information.
    """
    typer.echo(f"[DP] Segmenting transcripts from {input_dir}")
    typer.echo(f"Using order papers from {order_papers_dir}")
    typer.echo(f"Output file: {out}")
    
    # Load all motions from order papers
    motions = load_motions(order_papers_dir)
    typer.echo(f"Loaded {len(motions)} motions from order papers")
    
    # Process all sitting files
    sitting_files = list(input_dir.glob("*.html"))
    utterances = []
    
    for sitting_file in tqdm(sitting_files, desc="Processing sittings"):
        sitting_utterances = process_sitting(sitting_file, motions)
        utterances.extend(sitting_utterances)
    
    # Save utterances as JSONL
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        for utterance in utterances:
            f.write(json.dumps(utterance) + '\n')
    
    typer.echo(f"✓ Processed {len(utterances)} utterances from {len(sitting_files)} sittings")


def load_motions(order_papers_dir: Path) -> Dict[str, Any]:
    """Load all motions from order paper JSON files."""
    motions = {}
    
    for order_paper_file in order_papers_dir.glob("*.json"):
        with open(order_paper_file) as f:
            data = json.load(f)
            for motion in data.get('motions', []):
                motions[motion['motion_id']] = motion
    
    return motions


def process_sitting(sitting_file: Path, motions: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process a single sitting file and extract utterances."""
    with open(sitting_file) as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    
    utterances = []
    sitting_id = sitting_file.stem
    
    # Extract speaker turns (this is a simplified example)
    speaker_divs = soup.find_all('div', class_='speaker')
    
    for i, speaker_div in enumerate(speaker_divs):
        text = speaker_div.get_text().strip()
        
        # Parse speaker and utterance
        if ':' in text:
            speaker, utterance_text = text.split(':', 1)
            speaker = speaker.strip()
            utterance_text = utterance_text.strip()
        else:
            speaker = "Unknown"
            utterance_text = text
        
        # Generate timestamp (placeholder)
        timestamp = f"2023-01-01T{10 + i//10:02d}:{(i*2) % 60:02d}:00"
        
        # Link to motion (simplified heuristic)
        linked_motion = find_relevant_motion(utterance_text, motions)
        
        utterance = {
            "sitting_id": sitting_id,
            "utterance_id": f"{sitting_id}_{i:03d}",
            "speaker": speaker,
            "timestamp": timestamp,
            "text": utterance_text,
            "motion_id": linked_motion['motion_id'] if linked_motion else None,
            "motion_text": linked_motion['text'] if linked_motion else None,
        }
        
        utterances.append(utterance)
    
    return utterances


def find_relevant_motion(utterance_text: str, motions: Dict[str, Any]) -> Dict[str, Any]:
    """Find the most relevant motion for an utterance using simple heuristics."""
    # Simple keyword matching approach
    # In a real implementation, you might use more sophisticated NLP techniques
    
    best_motion = None
    best_score = 0
    
    utterance_words = set(utterance_text.lower().split())
    
    for motion_id, motion in motions.items():
        motion_words = set(motion['text'].lower().split())
        
        # Calculate simple word overlap score
        overlap = len(utterance_words & motion_words)
        score = overlap / len(motion_words) if motion_words else 0
        
        if score > best_score:
            best_score = score
            best_motion = motion
    
    return best_motion if best_score > 0.1 else None  # Threshold for relevance


if __name__ == "__main__":
    app()
