"""
Automated annotation using Gemma 3:270M via Ollama.

Usage:
    python -m src.label.auto_annotate --in data/interim/utterances_full.jsonl --out data/processed/auto_annotated.csv --model gemma3:270m
"""

import typer
from pathlib import Path
import json
import pandas as pd
import requests
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import random


app = typer.Typer()


@app.command()
def main(
    input_file: Path = typer.Option(..., "--in", help="Input JSONL file with utterances"),
    out: Path = typer.Option(..., help="Output CSV file for auto-annotated data"),
    model: str = typer.Option("gemma3:270m", help="Ollama model to use for annotation"),
    max_utterances: int = typer.Option(1000, help="Maximum number of utterances to annotate"),
    delay: float = typer.Option(0.5, help="Delay between API calls in seconds"),
    sample_motion_linked: bool = typer.Option(True, help="Prioritize utterances with linked motions"),
    ollama_url: str = typer.Option("http://localhost:11434", help="Ollama API URL"),
):
    """
    Automatically annotate utterances using Gemma 3:270M via Ollama.
    
    This creates relevance labels for utterances based on their relationship
    to associated parliamentary motions.
    """
    typer.echo(f"[Auto-Annotation] Using {model} via Ollama")
    typer.echo(f"Input: {input_file}")
    typer.echo(f"Output: {out}")
    typer.echo(f"Max utterances: {max_utterances}")
    
    # Check if Ollama is running
    if not check_ollama_connection(ollama_url):
        typer.echo("Error: Cannot connect to Ollama. Make sure it's running.")
        raise typer.Exit(1)
    
    # Load utterances
    utterances = load_utterances(input_file, max_utterances, sample_motion_linked)
    typer.echo(f"Loaded {len(utterances)} utterances for annotation")
    
    # Create annotation prompt template
    prompt_template = create_annotation_prompt()
    
    # Annotate utterances
    annotated_data = []
    failed_annotations = 0
    
    for i, utterance in enumerate(tqdm(utterances, desc="Annotating utterances")):
        try:
            # Create prompt for this utterance
            prompt = create_utterance_prompt(utterance, prompt_template)
            
            # Get annotation from Ollama
            annotation = get_ollama_annotation(prompt, model, ollama_url)
            
            # Parse annotation
            parsed_annotation = parse_annotation(annotation, utterance)
            
            # Add to results
            annotated_data.append(parsed_annotation)
            
            # Add delay to avoid overwhelming the API
            if delay > 0:
                time.sleep(delay)
                
        except Exception as e:
            typer.echo(f"Error annotating utterance {i}: {e}")
            failed_annotations += 1
            continue
    
    # Save results
    save_annotations(annotated_data, out)
    
    typer.echo(f"✓ Annotated {len(annotated_data)} utterances")
    if failed_annotations > 0:
        typer.echo(f"⚠ Failed to annotate {failed_annotations} utterances")
    
    # Print summary statistics
    print_annotation_summary(annotated_data)


def check_ollama_connection(ollama_url: str) -> bool:
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def load_utterances(input_file: Path, max_utterances: int, sample_motion_linked: bool) -> List[Dict[str, Any]]:
    """Load utterances from JSONL file, prioritizing motion-linked ones."""
    utterances = []
    
    with open(input_file, encoding='utf-8') as f:
        for line in f:
            utterances.append(json.loads(line.strip()))
    
    # Prioritize utterances with linked motions if requested
    if sample_motion_linked:
        motion_linked = [u for u in utterances if u.get('motion_id')]
        non_motion_linked = [u for u in utterances if not u.get('motion_id')]
        
        # Take all motion-linked utterances, then fill with non-motion-linked
        if len(motion_linked) >= max_utterances:
            utterances = random.sample(motion_linked, max_utterances)
        else:
            remaining = max_utterances - len(motion_linked)
            utterances = motion_linked + random.sample(non_motion_linked, min(remaining, len(non_motion_linked)))
    else:
        utterances = random.sample(utterances, min(max_utterances, len(utterances)))
    
    return utterances


def create_annotation_prompt() -> str:
    """Create the base prompt template for annotation."""
    return """You are an expert parliamentary analyst. Your task is to determine if a parliamentary utterance is relevant to a specific motion.

STRICT RELEVANCE CRITERIA:
- RELEVANT: The utterance MUST directly discuss, argue for/against, provide evidence for, or propose actions related to the SPECIFIC CONTENT of the motion
- NOT_RELEVANT: The utterance is procedural, a greeting, off-topic, a joke, unrelated to the motion, or only tangentially related

IMPORTANT: Be STRICT. Only label as RELEVANT if there is a clear, direct connection to the motion's specific content.

INSTRUCTIONS:
1. Read the motion text carefully - identify the specific topic/subject
2. Read the utterance text carefully - identify what it's actually about
3. Determine if the utterance directly relates to the motion's specific content
4. Respond with ONLY: RELEVANT or NOT_RELEVANT
5. Provide a brief explanation (1-2 sentences) for your decision

EXAMPLES:
Motion: "Adopt the Report on Budget Performance"
Utterance: "Madam Speaker, I support this motion because the budget shows good progress in infrastructure development."
Response: RELEVANT - The speaker explicitly supports the motion and provides reasoning related to budget performance.

Motion: "Adopt the Report on Budget Performance"
Utterance: "Madam Speaker, I rise to ask about the weather forecast for next week."
Response: NOT_RELEVANT - The utterance is completely unrelated to the budget performance report.

Motion: "Adopt the Report on Budget Performance"
Utterance: "Madam Speaker, does the Seconder wish to speak now or later?"
Response: NOT_RELEVANT - This is a procedural question about speaking order, not about budget performance.

Motion: "Adopt the Report on Budget Performance"
Utterance: "Madam Speaker, I believe the hon. Member was in the House when this was raised before."
Response: NOT_RELEVANT - This is a procedural comment about previous discussions, not about budget performance content.

Now analyze this case:"""


def create_utterance_prompt(utterance: Dict[str, Any], prompt_template: str) -> str:
    """Create a specific prompt for an utterance."""
    motion_text = utterance.get('motion_text', 'No motion text available')
    utterance_text = utterance.get('text', '')
    speaker = utterance.get('speaker', 'Unknown Speaker')
    
    prompt = f"""{prompt_template}

MOTION TEXT: {motion_text}

UTTERANCE:
Speaker: {speaker}
Text: {utterance_text}

ANALYSIS:"""
    
    return prompt


def get_ollama_annotation(prompt: str, model: str, ollama_url: str) -> str:
    """Get annotation from Ollama API."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,  # Low temperature for consistent results
            "top_p": 0.9,
            "max_tokens": 200
        }
    }
    
    response = requests.post(
        f"{ollama_url}/api/generate",
        json=payload,
        timeout=30
    )
    
    if response.status_code != 200:
        raise Exception(f"Ollama API error: {response.status_code}")
    
    result = response.json()
    return result.get('response', '').strip()


def parse_annotation(annotation: str, utterance: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the LLM annotation and create structured output."""
    # Clean the annotation text
    annotation_clean = annotation.strip().upper()
    
    # Extract label (RELEVANT or NOT_RELEVANT)
    label = "UNKNOWN"
    explanation = ""
    
    # Look for explicit labels
    if annotation_clean.startswith("RELEVANT"):
        label = "RELEVANT"
        # Extract explanation after RELEVANT
        parts = annotation.split("RELEVANT", 1)
        if len(parts) > 1:
            explanation = parts[1].strip()
    elif annotation_clean.startswith("NOT_RELEVANT"):
        label = "NOT_RELEVANT"
        # Extract explanation after NOT_RELEVANT
        parts = annotation.split("NOT_RELEVANT", 1)
        if len(parts) > 1:
            explanation = parts[1].strip()
    else:
        # Try to find labels anywhere in the text
        if "NOT_RELEVANT" in annotation_clean:
            label = "NOT_RELEVANT"
            # Find the last occurrence of NOT_RELEVANT
            last_idx = annotation.rfind("NOT_RELEVANT")
            explanation = annotation[last_idx + len("NOT_RELEVANT"):].strip()
        elif "RELEVANT" in annotation_clean:
            label = "RELEVANT"
            # Find the last occurrence of RELEVANT
            last_idx = annotation.rfind("RELEVANT")
            explanation = annotation[last_idx + len("RELEVANT"):].strip()
    
    # Clean up explanation
    explanation = explanation.replace("-", "").strip()
    if explanation.startswith(":"):
        explanation = explanation[1:].strip()
    
    # If explanation is too long, truncate it
    if len(explanation) > 500:
        explanation = explanation[:500] + "..."
    
    return {
        'sitting_id': utterance.get('sitting_id', ''),
        'utterance_id': utterance.get('utterance_id', ''),
        'speaker': utterance.get('speaker', ''),
        'timestamp': utterance.get('timestamp', ''),
        'text': utterance.get('text', ''),
        'motion_id': utterance.get('motion_id', ''),
        'motion_text': utterance.get('motion_text', ''),
        'motion_type': utterance.get('motion_type', ''),
        'sitting_date': utterance.get('sitting_date', ''),
        'session_info': utterance.get('session_info', {}),
        'word_count': utterance.get('word_count', 0),
        'char_count': utterance.get('char_count', 0),
        'llm_label': label,
        'llm_explanation': explanation,
        'llm_raw_response': annotation,
        'annotator': 'gemma3:270m',
        'confidence': 'auto',  # Could be enhanced with confidence scoring
        'notes': ''
    }


def save_annotations(annotated_data: List[Dict[str, Any]], out: Path) -> None:
    """Save annotated data to CSV file."""
    out.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(annotated_data)
    df.to_csv(out, index=False)
    
    typer.echo(f"Saved {len(annotated_data)} annotations to {out}")


def print_annotation_summary(annotated_data: List[Dict[str, Any]]) -> None:
    """Print summary statistics of the annotations."""
    if not annotated_data:
        return
    
    print("\n" + "="*60)
    print("AUTOMATED ANNOTATION SUMMARY")
    print("="*60)
    
    # Label distribution
    labels = [item['llm_label'] for item in annotated_data]
    relevant_count = labels.count('RELEVANT')
    not_relevant_count = labels.count('NOT_RELEVANT')
    unknown_count = labels.count('UNKNOWN')
    
    print(f"Total annotations: {len(annotated_data)}")
    print(f"RELEVANT: {relevant_count} ({relevant_count/len(annotated_data)*100:.1f}%)")
    print(f"NOT_RELEVANT: {not_relevant_count} ({not_relevant_count/len(annotated_data)*100:.1f}%)")
    print(f"UNKNOWN: {unknown_count} ({unknown_count/len(annotated_data)*100:.1f}%)")
    
    # Motion linkage statistics
    motion_linked = sum(1 for item in annotated_data if item.get('motion_id'))
    print(f"\nMotion-linked utterances: {motion_linked} ({motion_linked/len(annotated_data)*100:.1f}%)")
    
    # Speaker statistics
    speakers = {}
    for item in annotated_data:
        speaker = item['speaker']
        speakers[speaker] = speakers.get(speaker, 0) + 1
    
    print(f"Unique speakers: {len(speakers)}")
    print("Top 5 speakers:")
    for speaker, count in sorted(speakers.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {speaker}: {count} utterances")
    
    # Length statistics
    word_counts = [item['word_count'] for item in annotated_data]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
    print(f"\nAverage utterance length: {avg_words:.1f} words")
    
    print("="*60)


if __name__ == "__main__":
    app()
