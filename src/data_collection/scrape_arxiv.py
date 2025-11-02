# src/data_collection/scrape_arxiv.py
"""
Collects monthly submission counts from arXiv for cs.AI and cs.LG categories.
"""

import requests
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime, timedelta
from typing import List
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArxivScraper:
    """Scrape arXiv submission counts via API."""
    
    API_URL = "http://export.arxiv.org/api/query"
    
    CATEGORIES = ["cs.AI", "cs.LG"]
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_monthly_submissions(
        self, 
        category: str, 
        start_date: str = "2022-01-01",
        end_date: str = "2025-10-01"
    ) -> pd.DataFrame:
        """Get monthly submission counts for a category using submittedDate filter."""
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        records = []
        
        current = start
        while current < end:
            next_month = (current + timedelta(days=32)).replace(day=1)
            
            # Build query with category AND submittedDate filter
            # Format: cat:cs.AI AND submittedDate:[YYYYMMDD0000 TO YYYYMMDD2359]
            date_start = current.strftime('%Y%m%d') + "0000"
            date_end = (next_month - timedelta(days=1)).strftime('%Y%m%d') + "2359"
            
            query = f"cat:{category} AND submittedDate:[{date_start} TO {date_end}]"
            params = {
                'search_query': query,
                'start': 0,
                'max_results': 1  # We only need the total count
            }
            
            try:
                response = self.session.get(self.API_URL, params=params, timeout=30)
                response.raise_for_status()
                
                # Parse XML
                root = ET.fromstring(response.content)
                namespace = {
                    'atom': 'http://www.w3.org/2005/Atom',
                    'opensearch': 'http://a9.com/-/spec/opensearch/1.1/'
                }
                total_results = root.find('.//opensearch:totalResults', namespace)
                
                count = int(total_results.text) if total_results is not None else 0
                
                records.append({
                    'category': category,
                    'date': current,
                    'year_month': current.strftime('%Y-%m'),
                    'submission_count': count
                })
                
                logger.info(f"{category} {current.strftime('%Y-%m')}: {count} submissions")
                
            except Exception as e:
                logger.error(f"Failed to fetch {category} for {current.strftime('%Y-%m')}: {e}")
                raise RuntimeError(
                    f"CRITICAL: arXiv API request failed for {category} {current.strftime('%Y-%m')}!\n"
                    f"Error: {e}\n"
                    "Cannot proceed without real data."
                )
            
            time.sleep(3)  # arXiv rate limit (be nice!)
            current = next_month
        
        return pd.DataFrame(records)
    
    def scrape_all(self, start_date: str = "2022-01-01", end_date: str = "2025-10-01") -> pd.DataFrame:
        """Scrape all categories."""
        
        all_data = []
        
        for category in self.CATEGORIES:
            df = self.get_monthly_submissions(category, start_date, end_date)
            all_data.append(df)
        
        combined = pd.concat(all_data, ignore_index=True)
        combined.to_csv("data/raw/arxiv_submissions.csv", index=False)
        
        return combined


# Synthetic data generator
def generate_fallback_data(self):
        """
        REMOVED: No synthetic data generation
        All data must come from real arXiv API
        """
        raise NotImplementedError(
            "Synthetic data generation has been removed. "
            "Only real arXiv API data is used. "
            "Ensure you have internet access and arXiv API is reachable."
        )


if __name__ == "__main__":
    import os
    
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    scraper = ArxivScraper()
    
    print("⚠️  NO SYNTHETIC DATA - Only real arXiv API data")
    print("Scraping arXiv... (this requires internet connection)")
    
    df = scraper.scrape_all()
    
    if df is None or len(df) == 0:
        raise RuntimeError(
            "CRITICAL: arXiv scraping failed with no data returned!\n"
            "Possible causes:\n"
            "  1. No internet connection\n"
            "  2. arXiv API is down\n"
            "  3. Rate limiting issues\n"
            "Cannot proceed without real data."
        )
    
    if df['submission_count'].isna().sum() > len(df) * 0.5:
        raise ValueError(
            f"CRITICAL: Too many missing values ({df['submission_count'].isna().sum()} / {len(df)})!\n"
            "Data quality is insufficient for research purposes."
        )
    
    # Aggregate cs.AI + cs.LG
    aggregated = df.groupby('year_month')['submission_count'].sum().reset_index()
    aggregated.columns = ['year_month', 'total_submissions']
    
    aggregated.to_csv("data/processed/arxiv_monthly.csv", index=False)
    
    print(f"✓ Successfully collected {len(aggregated)} monthly arXiv records")
    print(aggregated.head())
