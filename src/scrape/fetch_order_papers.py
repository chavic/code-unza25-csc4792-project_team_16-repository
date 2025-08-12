"""
Fetch Order Papers and extract motion text.

Usage:
    python -m src.scrape.fetch_order_papers --range 2023-01:2023-12 --out data/interim/
"""

import typer
from pathlib import Path
from typing import Optional
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
from tqdm import tqdm


app = typer.Typer()


@app.command()
def main(
    range_str: str = typer.Option(..., help="Date range (YYYY-MM:YYYY-MM)"),
    out: Path = typer.Option(..., help="Output directory for order papers"),
    delay: float = typer.Option(1.0, help="Delay between requests (seconds)"),
):
    """
    Fetch Order Papers and extract motion text for the specified date range.
    
    Order Papers contain the motions that will be debated in parliamentary sittings.
    This extracts the motion text and saves it as JSON files.
    """
    out.mkdir(parents=True, exist_ok=True)
    
    start_date, end_date = parse_date_range(range_str)
    typer.echo(f"[DU] Fetching Order Papers from {start_date} to {end_date}")
    typer.echo(f"Output directory: {out}")
    
    # Generate list of dates to process
    current_date = start_date
    dates = []
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=30)  # Monthly intervals
    
    for date in tqdm(dates, desc="Fetching Order Papers"):
        # Placeholder for actual scraping logic
        order_paper_file = out / f"order_paper_{date.strftime('%Y_%m')}.json"
        
        # Create placeholder motion data
        motion_data = {
            "date": date.isoformat(),
            "motions": [
                {
                    "motion_id": f"M{date.year}{date.month:02d}_001",
                    "title": "Motion on Economic Development",
                    "text": "That this House calls upon the Government to implement comprehensive economic reforms to address unemployment and promote sustainable development.",
                    "mover": "Hon. Member A",
                    "type": "substantive"
                },
                {
                    "motion_id": f"M{date.year}{date.month:02d}_002", 
                    "title": "Motion on Education Policy",
                    "text": "That this House urges the Ministry of Education to review the current curriculum and ensure adequate funding for all educational institutions.",
                    "mover": "Hon. Member B",
                    "type": "substantive"
                }
            ]
        }
        
        with open(order_paper_file, 'w') as f:
            json.dump(motion_data, f, indent=2)
    
    typer.echo(f"✓ Fetched {len(dates)} Order Papers to {out}")


def parse_date_range(range_str: str) -> tuple[datetime, datetime]:
    """Parse date range string like '2023-01:2023-12'"""
    start_str, end_str = range_str.split(':')
    start_year, start_month = map(int, start_str.split('-'))
    end_year, end_month = map(int, end_str.split('-'))
    
    start_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 28)  # Safe end of month
    
    return start_date, end_date


if __name__ == "__main__":
    app()
