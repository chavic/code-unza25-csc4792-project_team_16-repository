"""
Fetch Order Papers and extract motion text.

Usage:
    python -m src.scrape.fetch_order_papers --out data/interim/
"""

import typer
from pathlib import Path
from typing import Optional, List
import requests
from bs4 import BeautifulSoup
import json
import hashlib
import time
import re
from datetime import datetime, timedelta
from tqdm import tqdm
import urllib3

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = typer.Typer()

# Base URLs for Order Papers
BASE_URL = "https://www.parliament.gov.zm"
ORDER_PAPERS_URL = f"{BASE_URL}/publications/order-paper-list"

# Headers for polite crawling
HEADERS = {
    'User-Agent': 'Parliament-Relevance-Classifier/1.0 (Educational Research; Team16@unza.zm)'
}


@app.command()
def main(
    out: Path = typer.Option(..., help="Output directory for order papers"),
    max_papers: int = typer.Option(100, help="Maximum number of order papers to scrape"),
    delay: float = typer.Option(1.0, help="Delay between requests (seconds)"),
    max_pages: int = typer.Option(10, help="Maximum pages to crawl from order papers index"),
):
    """
    Fetch Order Papers and extract motion text from the National Assembly website.
    
    Order Papers contain the motions that will be debated in parliamentary sittings.
    This extracts the motion text and saves it as JSON files.
    """
    out.mkdir(parents=True, exist_ok=True)
    
    typer.echo(f"[DU] Starting to scrape Order Papers...")
    typer.echo(f"Output directory: {out}")
    typer.echo(f"Target papers: {max_papers}")
    typer.echo(f"Delay between requests: {delay}s")
    
    # Collect order paper links from the index
    typer.echo("Collecting order paper links from index...")
    paper_links = scrape_order_papers_index(ORDER_PAPERS_URL, max_pages, delay)
    
    # Limit to requested number
    if len(paper_links) > max_papers:
        paper_links = paper_links[:max_papers]
    
    typer.echo(f"Found {len(paper_links)} order paper links to scrape")
    
    # Scrape individual order paper pages
    scraped_count = 0
    for link_info in tqdm(paper_links, desc="Scraping order papers"):
        try:
            success = scrape_order_paper_page(link_info, out, delay)
            if success:
                scraped_count += 1
        except Exception as e:
            typer.echo(f"Error scraping {link_info['url']}: {e}")
            continue
    
    typer.echo(f"✓ Successfully scraped {scraped_count}/{len(paper_links)} order papers to {out}")


def scrape_order_papers_index(base_url: str, max_pages: int, delay: float) -> List[dict]:
    """Scrape the order papers index pages to collect paper links."""
    paper_links = []
    
    for page in range(max_pages):
        try:
            if page == 0:
                url = base_url
            else:
                url = f"{base_url}?page={page}"
            
            typer.echo(f"  Scraping index page: {url}")
            
            response = requests.get(url, headers=HEADERS, timeout=10, verify=False)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all order paper links in the view content
            view_content = soup.find('div', class_='view-content')
            if not view_content:
                typer.echo(f"  No view content found on page {page}")
                break
            
            # Extract links from list items
            links = view_content.find_all('a', href=True)
            page_links = []
            
            for link in links:
                href = link.get('href')
                if href and href.startswith('/node/'):
                    full_url = BASE_URL + href
                    title = link.get_text().strip()
                    
                    # Extract date info if possible
                    date_str = extract_date_from_title(title)
                    
                    link_info = {
                        'url': full_url,
                        'node_id': href.split('/')[-1],
                        'title': title,
                        'date': date_str,
                        'source_index': base_url
                    }
                    page_links.append(link_info)
            
            paper_links.extend(page_links)
            typer.echo(f"  Found {len(page_links)} order paper links on page {page}")
            
            # Check if there are more pages
            pager = soup.find('ul', class_='pager')
            if not pager or not pager.find('a', title=lambda x: x and 'next' in x.lower()):
                typer.echo(f"  No more pages found after page {page}")
                break
            
            time.sleep(delay)
            
        except Exception as e:
            typer.echo(f"Error scraping index page {page}: {e}")
            break
    
    return paper_links


def scrape_order_paper_page(link_info: dict, out_dir: Path, delay: float) -> bool:
    """Scrape an individual order paper page and extract motion text."""
    try:
        url = link_info['url']
        node_id = link_info['node_id']
        
        response = requests.get(url, headers=HEADERS, timeout=15, verify=False)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract motion content from the page
        motion_data = extract_motions_from_page(soup, link_info)
        
        # Generate content hash for deduplication
        content_hash = hashlib.md5(response.content).hexdigest()[:8]
        
        # Create filename based on date and node ID
        date_str = link_info.get('date', 'unknown_date')
        filename = f"order_paper_{date_str}_{node_id}_{content_hash}.json"
        filepath = out_dir / filename
        
        # Save extracted motion data
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(motion_data, f, indent=2, ensure_ascii=False)
        
        # Also save raw HTML for reference
        html_filename = f"order_paper_{date_str}_{node_id}_{content_hash}.html"
        html_filepath = out_dir / html_filename
        with open(html_filepath, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        time.sleep(delay)
        return True
        
    except Exception as e:
        typer.echo(f"Failed to scrape {url}: {e}")
        return False


def extract_motions_from_page(soup: BeautifulSoup, link_info: dict) -> dict:
    """Extract motion text and metadata from an order paper page."""
    motion_data = {
        'url': link_info['url'],
        'node_id': link_info['node_id'],
        'title': link_info['title'],
        'date': link_info.get('date'),
        'scraped_at': datetime.now().isoformat(),
        'motions': [],
        'order_items': [],
        'session_info': None
    }
    
    # Find the main content area
    body_content = soup.find('div', class_='field-name-body')
    if not body_content:
        return motion_data
    
    # Extract session information
    session_info = extract_session_info(body_content)
    if session_info:
        motion_data['session_info'] = session_info
    
    # Extract motions and order items
    motions = extract_motion_text(body_content)
    motion_data['motions'] = motions
    
    # Extract order of the day items
    order_items = extract_order_items(body_content)
    motion_data['order_items'] = order_items
    
    return motion_data


def extract_session_info(content) -> dict:
    """Extract session and assembly information."""
    session_info = {}
    
    # Look for session information in the content
    text = content.get_text()
    
    # Extract assembly session (e.g., "Fourth Session of the Thirteenth Assembly")
    session_match = re.search(r'(First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|Eleventh|Twelfth|Thirteenth|Fourteenth|Fifteenth)\s+Session\s+of\s+the\s+(.*?)\s+Assembly', text, re.IGNORECASE)
    if session_match:
        session_info['session'] = session_match.group(1) + ' Session'
        session_info['assembly'] = session_match.group(2) + ' Assembly'
    
    # Extract date information
    date_match = re.search(r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s*(\d+\w*)\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s*(\d{4})', text, re.IGNORECASE)
    if date_match:
        session_info['sitting_date'] = f"{date_match.group(1)}, {date_match.group(2)} {date_match.group(3)}, {date_match.group(4)}"
    
    return session_info


def extract_motion_text(content) -> List[dict]:
    """Extract motion text from order paper content."""
    motions = []
    
    # Find all paragraphs that might contain motions
    paragraphs = content.find_all('p')
    
    motion_counter = 1
    for p in paragraphs:
        text = p.get_text().strip()
        
        # Look for motion patterns
        if is_motion_text(text):
            motion = {
                'motion_id': f"M_{motion_counter:03d}",
                'text': text,
                'type': classify_motion_type(text),
                'mover': extract_mover(text)
            }
            motions.append(motion)
            motion_counter += 1
    
    return motions


def extract_order_items(content) -> List[dict]:
    """Extract order of the day items."""
    order_items = []
    
    # Look for ordered lists or numbered items
    lists = content.find_all(['ol', 'ul'])
    
    for list_elem in lists:
        items = list_elem.find_all('li')
        for i, item in enumerate(items, 1):
            text = item.get_text().strip()
            if text:
                order_item = {
                    'item_number': i,
                    'text': text,
                    'type': classify_order_item_type(text)
                }
                order_items.append(order_item)
    
    return order_items


def is_motion_text(text: str) -> bool:
    """Determine if text is likely a motion."""
    motion_indicators = [
        'That this House',
        'That the House',
        'THAT this House',
        'THAT the House',
        'Motion',
        'MOTION',
        'Be it resolved',
        'Whereas'
    ]
    
    return any(indicator in text for indicator in motion_indicators) and len(text) > 50


def classify_motion_type(text: str) -> str:
    """Classify the type of motion."""
    text_lower = text.lower()
    
    if 'amendment' in text_lower:
        return 'amendment'
    elif 'budget' in text_lower or 'appropriation' in text_lower:
        return 'budget'
    elif 'committee' in text_lower:
        return 'committee'
    elif 'urgent' in text_lower:
        return 'urgent'
    else:
        return 'substantive'


def classify_order_item_type(text: str) -> str:
    """Classify the type of order item."""
    text_lower = text.lower()
    
    if 'question' in text_lower:
        return 'question'
    elif 'statement' in text_lower:
        return 'statement'
    elif 'bill' in text_lower:
        return 'bill'
    elif 'motion' in text_lower:
        return 'motion'
    else:
        return 'other'


def extract_mover(text: str) -> Optional[str]:
    """Extract the mover of the motion if mentioned."""
    # Look for patterns like "Hon. Member Name" or "Minister of..."
    mover_patterns = [
        r'Hon\.\s+([^,\n]+)',
        r'Minister\s+of\s+([^,\n]+)',
        r'Mr\.\s+([^,\n]+)',
        r'Mrs\.\s+([^,\n]+)',
        r'Ms\.\s+([^,\n]+)',
        r'Dr\.\s+([^,\n]+)'
    ]
    
    for pattern in mover_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    
    return None


def extract_date_from_title(title: str) -> str:
    """Extract date from order paper title and convert to YYYY-MM-DD format."""
    try:
        # Handle formats like "Tuesday, 15th July, 2025"
        import re
        
        # Remove day of week
        title = re.sub(r'^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s*', '', title)
        
        # Remove ordinal suffixes (1st, 2nd, 3rd, 4th, etc.)
        title = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', title)
        
        # Try to parse date
        from datetime import datetime
        
        # Common formats
        formats = [
            "%d %B, %Y",      # "15 July, 2025"
            "%d %B %Y",       # "15 July 2025"
            "%d/%m/%Y",       # "15/07/2025"
            "%Y-%m-%d",       # "2025-07-15"
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(title.strip(), fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        
        # If parsing fails, return a sanitized version
        sanitized = re.sub(r'[^\w\-]', '_', title.strip())[:20]
        return sanitized
        
    except Exception:
        return "unknown_date"


if __name__ == "__main__":
    app()
