# src/data_collection/scrape_stocks.py
"""
Downloads NVIDIA (NVDA) stock prices as proxy for AI investment sentiment.
Uses Alpha Vantage API (more reliable than yfinance/Yahoo Finance).
"""

import pandas as pd
import requests
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_nvidia_stock_alpha_vantage(
    start_date: str = "2022-01-01",
    end_date: str = "2025-10-31",
    config_path: str = "config.yaml"
) -> pd.DataFrame:
    """
    Download NVIDIA stock data using Alpha Vantage API.
    More reliable than yfinance with no rate limiting issues.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        config_path: Path to config.yaml with API key
    """
    
    logger.info(f"Downloading NVDA stock from Alpha Vantage ({start_date} to {end_date})")
    
    # Load API key from secrets.toml
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils import get_api_key
    
    api_key = get_api_key('alpha_vantage')
    
    # Alpha Vantage API endpoint
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'TIME_SERIES_MONTHLY',
        'symbol': 'NVDA',
        'apikey': api_key,
        'outputsize': 'full'  # Get all available data
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Check for API errors
        if 'Error Message' in data:
            raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
        
        if 'Note' in data:
            raise ValueError(f"Alpha Vantage rate limit: {data['Note']}")
        
        if 'Monthly Time Series' not in data:
            raise ValueError(f"Unexpected API response: {list(data.keys())}")
        
        # Parse time series data
        records = []
        time_series = data['Monthly Time Series']
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        for date_str, values in time_series.items():
            date = pd.to_datetime(date_str)
            
            # Filter by date range
            if date >= start_dt and date <= end_dt:
                records.append({
                    'date': date,
                    'year_month': date.to_period('M'),
                    'nvda_price': float(values['4. close']),
                    'nvda_volume': int(values['5. volume'])
                })
        
        if not records:
            raise ValueError(f"No data found in date range {start_date} to {end_date}")
        
        # Create DataFrame
        df = pd.DataFrame(records).sort_values('date')
        
        # Save to raw data
        output_path = Path("data/raw/nvidia_stock.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        logger.info(f"✓ Downloaded {len(df)} months of NVDA stock data")
        logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"  Price range: ${df['nvda_price'].min():.2f} - ${df['nvda_price'].max():.2f}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {e}")
        raise RuntimeError(
            f"CRITICAL: Cannot download NVIDIA stock data from Alpha Vantage: {e}\n"
            "Check internet connection and API key."
        )
    
    except Exception as e:
        logger.error(f"Failed to download stock data: {e}")
        raise RuntimeError(
            f"CRITICAL: Cannot download NVIDIA stock data: {e}\n"
            "This might be due to API rate limiting or invalid API key.\n"
            "Cannot proceed without real data."
        )


if __name__ == "__main__":
    import os
    
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    print("⚠️  NO SYNTHETIC DATA - Only real Alpha Vantage data")
    print("Downloading NVIDIA stock data... (this requires internet connection)")
    
    df = download_nvidia_stock_alpha_vantage()
    
    if df is None or df.empty:
        raise RuntimeError(
            "CRITICAL: Alpha Vantage download failed with no data!\n"
            "Possible causes:\n"
            "  1. No internet connection\n"
            "  2. Invalid API key\n"
            "  3. API rate limit exceeded\n"
            "Cannot proceed without real data."
        )
    
    df.to_csv("data/processed/nvidia_monthly.csv", index=False)
    
    print(f"\n✓ Successfully collected {len(df)} monthly stock records")
    print("\nSample data:")
    print(df.head(10))
