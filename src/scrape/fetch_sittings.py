"""
Fetch parliamentary sitting transcripts.

Usage:
    python -m src.scrape.fetch_sittings --out data/raw/
"""

import typer
from pathlib import Path
from typing import Optional, List, Tuple
import requests
from bs4 import BeautifulSoup
import time
import json
import hashlib
from datetime import datetime
from tqdm import tqdm
import urllib3

# Suppress SSL warnings for the Parliament website
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


app = typer.Typer()

# Base URLs for the Parliament website
BASE_URL = "https://www.parliament.gov.zm"
DEBATES_LIST_URL = f"{BASE_URL}/publications/debates-list"
DEBATES_ALT_URL = f"{BASE_URL}/publications/debates-proceedings"

# Headers for polite crawling
HEADERS = {
    'User-Agent': 'Parliament-Relevance-Classifier/1.0 (Educational Research; Team16@unza.zm)'
}


@app.command()
def main(
    out: Path = typer.Option(..., help="Output directory for raw sitting files"),
    num_sittings: int = typer.Option(10, help="Number of sittings to scrape"),
    delay: float = typer.Option(1.0, help="Delay between requests (seconds)"),
    start_year: int = typer.Option(2023, help="Starting year for scraping"),
    max_pages: int = typer.Option(5, help="Maximum pages to crawl from each index"),
    use_alt_index: bool = typer.Option(False, help="Also crawl alternate debates index"),
):
    """
    Fetch parliamentary sitting transcripts from the National Assembly website.
    
    This will scrape debate/proceeding transcripts and save them as HTML files
    in the specified output directory.
    """
    out.mkdir(parents=True, exist_ok=True)
    
    typer.echo(f"[BU] Starting to scrape parliamentary sittings...")
    typer.echo(f"Output directory: {out}")
    typer.echo(f"Target sittings: {num_sittings}")
    typer.echo(f"Delay between requests: {delay}s")
    
    # Collect sitting links from the main debates index
    sitting_links = []
    
    typer.echo("Collecting sitting links from main debates index...")
    main_links = scrape_debates_index(DEBATES_LIST_URL, max_pages, delay)
    sitting_links.extend(main_links)
    
    if use_alt_index:
        typer.echo("Collecting sitting links from alternate debates index...")
        alt_links = scrape_debates_index(DEBATES_ALT_URL, max_pages, delay)
        sitting_links.extend(alt_links)
    
    # Remove duplicates based on URL and limit to requested number
    seen_urls = set()
    unique_links = []
    for link in sitting_links:
        if link['url'] not in seen_urls:
            seen_urls.add(link['url'])
            unique_links.append(link)
    
    if len(unique_links) > num_sittings:
        unique_links = unique_links[:num_sittings]
    
    typer.echo(f"Found {len(unique_links)} unique sitting links to scrape")
    
    # Scrape individual sitting pages
    scraped_count = 0
    for link_info in tqdm(unique_links, desc="Scraping individual sittings"):
        try:
            success = scrape_sitting_page(link_info, out, delay)
            if success:
                scraped_count += 1
        except Exception as e:
            typer.echo(f"Error scraping {link_info['url']}: {e}")
            continue
    
    typer.echo(f"✓ Successfully scraped {scraped_count}/{len(unique_links)} sittings to {out}")


def scrape_debates_index(base_url: str, max_pages: int, delay: float) -> List[dict]:
    """Scrape the debates index pages to collect sitting links."""
    sitting_links = []
    
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
            
            # Find all sitting links in the view content
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
            
            sitting_links.extend(page_links)
            typer.echo(f"  Found {len(page_links)} sitting links on page {page}")
            
            # Check if there are more pages
            pager = soup.find('ul', class_='pager')
            if not pager or not pager.find('a', title=lambda x: x and 'next' in x.lower()):
                typer.echo(f"  No more pages found after page {page}")
                break
            
            time.sleep(delay)
            
        except Exception as e:
            typer.echo(f"Error scraping index page {page}: {e}")
            break
    
    return sitting_links


def scrape_sitting_page(link_info: dict, out_dir: Path, delay: float) -> bool:
    """Scrape an individual sitting page and save to file."""
    try:
        url = link_info['url']
        node_id = link_info['node_id']
        
        response = requests.get(url, headers=HEADERS, timeout=15, verify=False)
        response.raise_for_status()
        
        # Generate content hash for deduplication
        content_hash = hashlib.md5(response.content).hexdigest()[:8]
        
        # Create filename based on date and node ID
        date_str = link_info.get('date', 'unknown_date')
        filename = f"sitting_{date_str}_{node_id}_{content_hash}.html"
        filepath = out_dir / filename
        
        # Save raw HTML
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        # Also save metadata
        metadata = {
            'url': url,
            'node_id': node_id,
            'title': link_info['title'],
            'date': link_info.get('date'),
            'source_index': link_info.get('source_index'),
            'content_hash': content_hash,
            'scraped_at': datetime.now().isoformat(),
            'filename': filename
        }
        
        metadata_file = out_dir / f"sitting_{date_str}_{node_id}_{content_hash}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        time.sleep(delay)
        return True
        
    except Exception as e:
        typer.echo(f"Failed to scrape {url}: {e}")
        return False


def extract_date_from_title(title: str) -> str:
    """Extract date from sitting title and convert to YYYY-MM-DD format."""
    try:
        # Handle formats like "Tuesday, 15th July, 2025"
        # Remove day of week and ordinal suffixes
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
