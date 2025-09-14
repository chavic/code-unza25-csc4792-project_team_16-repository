"""
Create sitting-wise train/validation/test splits for the dataset.

Usage:
    python -m src.parse.split_data --in data/interim/utterances.jsonl --out data/processed/ --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
"""

import typer
from pathlib import Path
import json
import random
from typing import List, Dict, Any
from collections import defaultdict


app = typer.Typer()


@app.command()
def main(
    input_file: Path = typer.Option(..., "--in", help="Input JSONL file with utterances"),
    out_dir: Path = typer.Option(..., "--out", help="Output directory for split files"),
    train_ratio: float = typer.Option(0.7, help="Ratio of sittings for training"),
    val_ratio: float = typer.Option(0.15, help="Ratio of sittings for validation"),
    test_ratio: float = typer.Option(0.15, help="Ratio of sittings for testing"),
    seed: int = typer.Option(42, help="Random seed for reproducible splits"),
):
    """
    Create sitting-wise train/validation/test splits.
    
    This ensures that utterances from the same sitting stay together in the same split,
    preventing data leakage between training and evaluation.
    """
    typer.echo(f"[DP] Creating sitting-wise data splits from {input_file}")
    typer.echo(f"Split ratios: Train={train_ratio:.1%}, Val={val_ratio:.1%}, Test={test_ratio:.1%}")
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        typer.echo(f"Error: Ratios must sum to 1.0, got {total_ratio}")
        raise typer.Exit(1)
    
    random.seed(seed)
    
    # Load utterances and group by sitting
    sitting_utterances = defaultdict(list)
    total_utterances = 0
    
    typer.echo("Loading utterances...")
    with open(input_file, encoding='utf-8') as f:
        for line in f:
            utterance = json.loads(line.strip())
            sitting_id = utterance['sitting_id']
            sitting_utterances[sitting_id].append(utterance)
            total_utterances += 1
    
    typer.echo(f"Loaded {total_utterances} utterances from {len(sitting_utterances)} sittings")
    
    # Create sitting-wise splits
    sitting_ids = list(sitting_utterances.keys())
    random.shuffle(sitting_ids)
    
    n_sittings = len(sitting_ids)
    n_train = int(n_sittings * train_ratio)
    n_val = int(n_sittings * val_ratio)
    n_test = n_sittings - n_train - n_val  # Remaining for test
    
    train_sittings = sitting_ids[:n_train]
    val_sittings = sitting_ids[n_train:n_train + n_val]
    test_sittings = sitting_ids[n_train + n_val:]
    
    typer.echo(f"Split sizes:")
    typer.echo(f"  Train: {n_train} sittings")
    typer.echo(f"  Validation: {n_val} sittings")
    typer.echo(f"  Test: {n_test} sittings")
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Write splits
    splits = {
        'train': train_sittings,
        'val': val_sittings,
        'test': test_sittings
    }
    
    for split_name, split_sittings in splits.items():
        output_file = out_dir / f"{split_name}.jsonl"
        split_utterances = []
        
        for sitting_id in split_sittings:
            split_utterances.extend(sitting_utterances[sitting_id])
        
        # Shuffle utterances within each split
        random.shuffle(split_utterances)
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            for utterance in split_utterances:
                f.write(json.dumps(utterance, ensure_ascii=False) + '\n')
        
        typer.echo(f"  {split_name.capitalize()}: {len(split_utterances)} utterances")
    
    # Create metadata file
    metadata = {
        'total_utterances': total_utterances,
        'total_sittings': n_sittings,
        'split_ratios': {
            'train': train_ratio,
            'val': val_ratio,
            'test': test_ratio
        },
        'split_sizes': {
            'train_sittings': n_train,
            'val_sittings': n_val,
            'test_sittings': n_test
        },
        'split_utterances': {
            'train': len([u for sitting in train_sittings for u in sitting_utterances[sitting]]),
            'val': len([u for sitting in val_sittings for u in sitting_utterances[sitting]]),
            'test': len([u for sitting in test_sittings for u in sitting_utterances[sitting]])
        },
        'random_seed': seed
    }
    
    metadata_file = out_dir / "split_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    typer.echo(f"✓ Created data splits in {out_dir}")
    typer.echo(f"✓ Saved metadata to {metadata_file}")
    
    # Print summary statistics
    print_split_summary(sitting_utterances, splits)


def print_split_summary(sitting_utterances: Dict[str, List[Dict]], splits: Dict[str, List[str]]) -> None:
    """Print detailed summary statistics for the splits."""
    print("\n" + "="*60)
    print("DATA SPLIT SUMMARY")
    print("="*60)
    
    for split_name, split_sittings in splits.items():
        split_utterances = [u for sitting in split_sittings for u in sitting_utterances[sitting]]
        
        print(f"\n{split_name.upper()} SPLIT:")
        print(f"  Sittings: {len(split_sittings)}")
        print(f"  Utterances: {len(split_utterances)}")
        
        # Speaker statistics
        speakers = {}
        for utt in split_utterances:
            speaker = utt['speaker']
            speakers[speaker] = speakers.get(speaker, 0) + 1
        
        print(f"  Unique speakers: {len(speakers)}")
        
        # Motion linkage statistics
        linked_count = sum(1 for utt in split_utterances if utt.get('motion_id'))
        print(f"  Utterances with motions: {linked_count} ({linked_count/len(split_utterances)*100:.1f}%)")
        
        # Length statistics
        word_counts = [utt['word_count'] for utt in split_utterances]
        avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
        print(f"  Average words per utterance: {avg_words:.1f}")
        
        # Date range
        dates = [utt['sitting_date'] for utt in split_utterances if utt.get('sitting_date') != 'unknown_date']
        if dates:
            dates.sort()
            print(f"  Date range: {dates[0]} to {dates[-1]}")
    
    print("="*60)


if __name__ == "__main__":
    app()
