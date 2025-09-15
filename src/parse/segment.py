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
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple


app = typer.Typer()


@app.command()
def main(
    input_dir: Path = typer.Option(..., "--in", help="Input directory with raw sitting files"),
    order_papers_dir: Path = typer.Option(..., help="Directory with order papers"),
    out: Path = typer.Option(..., help="Output JSONL file for utterances"),
    max_files: int = typer.Option(None, help="Maximum number of files to process (for testing)"),
):
    """
    Segment parliamentary transcripts into speaker utterances and link to motions.
    
    This processes the raw HTML files from scraping and creates structured
    utterances with speaker, timestamp, text, and linked motion information.
    """
    typer.echo(f"[DP] Segmenting transcripts from {input_dir}")
    typer.echo(f"Using order papers from {order_papers_dir}")
    typer.echo(f"Output file: {out}")
    
    # Load all motions from order papers (indexed by date)
    motions_by_date = load_motions_by_date(order_papers_dir)
    typer.echo(f"Loaded motions for {len(motions_by_date)} dates from order papers")
    
    # Process all sitting files
    sitting_files = list(input_dir.glob("sitting_*.html"))
    if max_files:
        sitting_files = sitting_files[:max_files]
        typer.echo(f"Processing only first {max_files} files for testing")
    
    utterances = []
    processed_files = 0
    failed_files = 0
    
    for sitting_file in tqdm(sitting_files, desc="Processing sittings"):
        try:
            sitting_utterances = process_sitting(sitting_file, motions_by_date)
            utterances.extend(sitting_utterances)
            processed_files += 1
        except Exception as e:
            typer.echo(f"Error processing {sitting_file}: {e}")
            failed_files += 1
            continue
    
    # Save utterances as JSONL
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        for utterance in utterances:
            f.write(json.dumps(utterance, ensure_ascii=False) + '\n')
    
    typer.echo(f"✓ Processed {len(utterances)} utterances from {processed_files} sittings")
    if failed_files > 0:
        typer.echo(f"⚠ Failed to process {failed_files} files")
    
    # Print summary statistics
    print_summary_stats(utterances)


def load_motions_by_date(order_papers_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load all motions from order paper JSON files, indexed by date."""
    motions_by_date = {}
    
    for order_paper_file in order_papers_dir.glob("order_paper_*.json"):
        try:
            with open(order_paper_file, encoding='utf-8') as f:
                data = json.load(f)
                date = data.get('date')
                if date and data.get('motions'):
                    motions_by_date[date] = data['motions']
        except Exception as e:
            continue
    
    return motions_by_date


def process_sitting(sitting_file: Path, motions_by_date: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Process a single sitting file and extract utterances."""
    with open(sitting_file, encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    
    utterances = []
    sitting_id = sitting_file.stem
    
    # Extract date from filename or metadata
    sitting_date = extract_sitting_date(sitting_file, soup)
    
    # Extract session info and timestamp
    session_info = extract_session_info(soup)
    base_timestamp = extract_base_timestamp(soup)
    
    # Find corresponding motions for this date
    sitting_motions = motions_by_date.get(sitting_date, [])
    
    # Find the main content area
    body_content = soup.find('div', class_='field-name-body')
    if not body_content:
        return utterances
    
    # Extract all paragraphs
    paragraphs = body_content.find_all('p')
    
    # Process paragraphs to find speaker utterances
    utterance_count = 0
    current_speaker = None
    current_utterance_parts = []
    
    for p in paragraphs:
        # Check if this paragraph contains a speaker
        speaker_info = extract_speaker_from_paragraph(p)
        
        if speaker_info:
            # Save previous utterance if exists
            if current_speaker and current_utterance_parts:
                utterance = create_utterance(
                    sitting_id, utterance_count, current_speaker, 
                    current_utterance_parts, base_timestamp, 
                    sitting_motions, session_info, sitting_date
                )
                if utterance:
                    utterances.append(utterance)
                    utterance_count += 1
            
            # Start new utterance
            current_speaker = speaker_info['speaker']
            current_utterance_parts = [speaker_info['text']] if speaker_info['text'] else []
            
        else:
            # Continue current utterance or skip if no current speaker
            text = extract_clean_text(p)
            if text and current_speaker:
                current_utterance_parts.append(text)
    
    # Don't forget the last utterance
    if current_speaker and current_utterance_parts:
        utterance = create_utterance(
            sitting_id, utterance_count, current_speaker, 
            current_utterance_parts, base_timestamp, 
            sitting_motions, session_info, sitting_date
        )
        if utterance:
            utterances.append(utterance)
    
    return utterances


def extract_sitting_date(sitting_file: Path, soup: BeautifulSoup) -> str:
    """Extract the sitting date from filename or HTML content."""
    # Try to extract from filename first (format: sitting_YYYY-MM-DD_...)
    filename = sitting_file.stem
    date_match = re.search(r'sitting_(\d{4}-\d{2}-\d{2})', filename)
    if date_match:
        return date_match.group(1)
    
    # Try to extract from HTML content
    title_elem = soup.find('title')
    if title_elem:
        title_text = title_elem.get_text()
        # Look for date patterns in title
        date_patterns = [
            r'(\d{1,2})\w*\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s+(\d{4})',
            r'(\d{4}-\d{2}-\d{2})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, title_text, re.IGNORECASE)
            if match:
                if len(match.groups()) == 3:
                    # Convert month name to number
                    month_names = {
                        'january': '01', 'february': '02', 'march': '03', 'april': '04',
                        'may': '05', 'june': '06', 'july': '07', 'august': '08',
                        'september': '09', 'october': '10', 'november': '11', 'december': '12'
                    }
                    day = match.group(1).zfill(2)
                    month = month_names.get(match.group(2).lower(), '01')
                    year = match.group(3)
                    return f"{year}-{month}-{day}"
                else:
                    return match.group(1)
    
    return "unknown_date"


def extract_session_info(soup: BeautifulSoup) -> Dict[str, str]:
    """Extract session and assembly information."""
    session_info = {}
    
    # Look for session information in the content
    body_content = soup.find('div', class_='field-name-body')
    if body_content:
        text = body_content.get_text()
        
        # Extract assembly session
        session_match = re.search(r'(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|Eleventh|Twelfth|Thirteenth|Fourteenth|Fifteenth)\s+Session\s+of\s+the\s+(.*?)\s+Assembly', text, re.IGNORECASE)
        if session_match:
            session_info['session'] = session_match.group(1) + ' Session'
            session_info['assembly'] = session_match.group(2) + ' Assembly'
    
    return session_info


def extract_base_timestamp(soup: BeautifulSoup) -> str:
    """Extract the base timestamp when the House met."""
    body_content = soup.find('div', class_='field-name-body')
    if body_content:
        text = body_content.get_text()
        
        # Look for "The House met at XXXX hours"
        time_match = re.search(r'The House met at\s*(\d{4})\s*hours', text, re.IGNORECASE)
        if time_match:
            time_str = time_match.group(1)
            hour = int(time_str[:2])
            minute = int(time_str[2:])
            return f"{hour:02d}:{minute:02d}:00"
    
    return "14:30:00"  # Default parliamentary time


def extract_speaker_from_paragraph(p) -> Optional[Dict[str, str]]:
    """Extract speaker information from a paragraph."""
    # Look for <strong> tags that indicate speakers
    strong_tags = p.find_all('strong')
    
    for strong in strong_tags:
        strong_text = strong.get_text().strip()
        
        # Check if this looks like a speaker (ends with colon)
        if strong_text.endswith(':'):
            speaker_name = strong_text[:-1].strip()
            
            # Get the remaining text after the speaker name
            remaining_text = ""
            
            # Get all text after this strong tag in the same paragraph
            for sibling in strong.next_siblings:
                if hasattr(sibling, 'get_text'):
                    remaining_text += sibling.get_text()
                else:
                    remaining_text += str(sibling)
            
            return {
                'speaker': clean_speaker_name(speaker_name),
                'text': remaining_text.strip()
            }
    
    return None


def clean_speaker_name(speaker: str) -> str:
    """Clean and standardize speaker names."""
    # Remove extra whitespace and normalize
    speaker = re.sub(r'\s+', ' ', speaker.strip())
    
    # Handle common variations
    speaker_mappings = {
        'Madam Speaker': 'Madam Speaker',
        'Mr Speaker': 'Mr Speaker', 
        'The Speaker': 'The Speaker',
        'Hon. Members': 'Hon. Members',
        'Hon Members': 'Hon. Members'
    }
    
    return speaker_mappings.get(speaker, speaker)


def extract_clean_text(p) -> str:
    """Extract clean text from a paragraph, removing HTML formatting."""
    if not p:
        return ""
    
    # Get text content
    text = p.get_text()
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove empty lines and very short text
    if len(text) < 3:
        return ""
    
    # Skip section headers and procedural markers
    skip_patterns = [
        r'^[A-Z\s]+$',  # All caps headers
        r'^\d+\.\s*$',  # Just numbers
        r'^_+$',        # Just underscores
        r'^PRAYER$',
        r'^NATIONAL ANTHEM$',
        r'^\[.*\]$'     # Bracketed procedural notes
    ]
    
    for pattern in skip_patterns:
        if re.match(pattern, text):
            return ""
    
    return text


def create_utterance(sitting_id: str, utterance_count: int, speaker: str, 
                    utterance_parts: List[str], base_timestamp: str,
                    motions: List[Dict[str, Any]], session_info: Dict[str, str],
                    sitting_date: str) -> Optional[Dict[str, Any]]:
    """Create a structured utterance object."""
    
    # Combine utterance parts
    full_text = ' '.join(utterance_parts).strip()
    
    # Skip very short utterances
    if len(full_text) < 10:
        return None
    
    # Generate timestamp (increment from base time)
    base_hour, base_minute, _ = base_timestamp.split(':')
    total_minutes = int(base_hour) * 60 + int(base_minute) + utterance_count * 2
    new_hour = (total_minutes // 60) % 24
    new_minute = total_minutes % 60
    timestamp = f"{new_hour:02d}:{new_minute:02d}:00"
    
    # Find most relevant motion
    linked_motion = find_most_relevant_motion(full_text, motions)
        
    utterance = {
        "sitting_id": sitting_id,
        "utterance_id": f"{sitting_id}_{utterance_count:04d}",
        "speaker": speaker,
        "timestamp": timestamp,
        "text": full_text,
        "sitting_date": sitting_date,
        "session_info": session_info,
        "motion_id": linked_motion.get('motion_id') if linked_motion else None,
        "motion_text": linked_motion.get('text') if linked_motion else None,
        "motion_type": linked_motion.get('type') if linked_motion else None,
        "word_count": len(full_text.split()),
        "char_count": len(full_text)
    }
    
    return utterance


def find_most_relevant_motion(utterance_text: str, motions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Find the most relevant motion for an utterance using keyword overlap."""
    if not motions:
        return None
    
    best_motion = None
    best_score = 0
    
    utterance_words = set(utterance_text.lower().split())
    
    for motion in motions:
        if not motion.get('text'):
            continue
            
        motion_words = set(motion['text'].lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(utterance_words & motion_words)
        union = len(utterance_words | motion_words)
        
        jaccard_score = intersection / union if union > 0 else 0
        
        # Boost score for longer overlaps
        overlap_boost = intersection / 10.0
        final_score = jaccard_score + overlap_boost
        
        if final_score > best_score:
            best_score = final_score
            best_motion = motion
    
    # Return motion only if score is above threshold
    return best_motion if best_score > 0.05 else None


def print_summary_stats(utterances: List[Dict[str, Any]]) -> None:
    """Print summary statistics about the processed utterances."""
    if not utterances:
        return
    
    print("\n" + "="*50)
    print("UTTERANCE PROCESSING SUMMARY")
    print("="*50)
    
    print(f"Total utterances: {len(utterances)}")
    
    # Speaker statistics
    speakers = {}
    for utt in utterances:
        speaker = utt['speaker']
        speakers[speaker] = speakers.get(speaker, 0) + 1
    
    print(f"Unique speakers: {len(speakers)}")
    print("Top 10 speakers:")
    for speaker, count in sorted(speakers.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {speaker}: {count} utterances")
    
    # Length statistics
    word_counts = [utt['word_count'] for utt in utterances]
    print(f"\nUtterance length statistics:")
    print(f"  Average words: {sum(word_counts) / len(word_counts):.1f}")
    print(f"  Min words: {min(word_counts)}")
    print(f"  Max words: {max(word_counts)}")
    
    # Motion linkage statistics
    linked_count = sum(1 for utt in utterances if utt['motion_id'])
    print(f"\nMotion linkage:")
    print(f"  Utterances linked to motions: {linked_count} ({linked_count/len(utterances)*100:.1f}%)")
    print(f"  Utterances without motions: {len(utterances) - linked_count}")
    
    print("="*50)


if __name__ == "__main__":
    app()
