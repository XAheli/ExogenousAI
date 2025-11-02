"""
Scrapes policy events from Google News using SERP API
Collects AI policy announcements, export controls, regulations, etc.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import yaml
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolicyEventScraper:
    """Scrape AI policy events using SERP API (Google News)"""
    
    SERP_API_URL = "https://serpapi.com/search.json"
    
    # Search queries for different types of policy events
    SEARCH_QUERIES = {
        'export_control': [
            # Broader, more conceptual queries
            'AI semiconductor export controls',
            'US-China tech war chips',
            'geopolitics of AI compute',
            
            # Specific policies and agencies
            'BIS Entity List AI',
            'foreign direct product rule AI chips',
            'US export controls high-performance computing',
            
            # Key companies and technologies
            'NVIDIA A100 H800 export restrictions',
            'ASML EUV lithography export ban',
            'TSMC export controls',
            
            # Impact and response
            'China AI chip self-sufficiency',
            'smuggling AI chips China',
            'global AI chip supply chain disruption',
            
            # 2025 Targeted Updates
            'BIS Entity List updates 2025 China AI chips',
            'Nvidia AMD China AI chip export restrictions 2025',
            'ASML China EUV DUV lithography export controls 2025',
            'China rare earth export controls semiconductors 2025',
            'AI chip smuggling China 2025 arrests'
        ],
        
        'compute_governance': [
            # Key policy drivers
            'Biden Executive Order AI compute reporting',
            'White House AI Council compute',
            
            # Technical thresholds and metrics
            'AI model training compute threshold',
            'large AI model FLOPs reporting',
            '10^26 FLOPs AI model regulation',
            
            # Implementation and enforcement
            'cloud provider AI model registry',
            'AI compute cluster transparency',
            'NIST AI safety and security guidelines',
            
            # Broader concepts and risks
            'preventing AI compute misuse',
            'AI compute governance framework',
            'international AI compute standards',
            
            # 2025 Targeted Updates
            '10^26 FLOP AI model regulation 2025',
            'NIST AI Risk Management Framework 2025 updates',
            'AI compute cluster governance CalCompute 2025',
            'AI model registry requirements cloud providers 2025'
        ],
        
        'regulation': [
            # Jurisdiction-specific laws and implementation
            'EU AI Act compliance',
            'EU AI Act high-risk systems',
            'China generative AI measures',
            'US AI regulation bill progress',
            'UK AI Bill',
            
            # Specific regulatory concepts
            'algorithmic transparency law',
            'AI auditing requirements',
            'deepfake regulation',
            'autonomous vehicle AI policy',
            
            # Standards and enforcement bodies
            'NIST AI Risk Management Framework',
            'AI Act enforcement mechanisms',
            'FTC AI regulation',
            
            # Broader governance
            'sector-specific AI regulation healthcare',
            'AI copyright law',
            
            # 2025 Targeted Updates
            'EU AI Act enforcement 2025 high-risk GPAI',
            'China generative AI labeling measures 2025',
            'UK AI Bill 2026 delay AI Safety Institute',
            'California AI safety bill SB 53 2025',
            'deepfake regulation 2025 TAKE IT DOWN Act',
            'AI audit requirements 2025 FTC',
            'autonomous vehicle AI regulation 2025',
            'AI copyright law 2025 Anthropic'
        ],
        
        'international_agreement': [
            # Specific named agreements and declarations
            'G7 Hiroshima AI Process',
            'Bletchley Declaration AI safety',
            'US-EU Trade and Technology Council AI',
            
            # Key international forums and initiatives
            'OECD AI Principles',
            'UN Global AI Compact',
            'Global Partnership on AI (GPAI)',
            'UK AI Safety Institute',
            
            # Thematic goals of cooperation
            'international AI safety standards',
            'frontier AI risk management',
            'US-EU AI policy alignment',
            'China-US AI dialogue',
            
            # Practical collaboration
            'AI incident reporting network',
            'shared AI research testbeds',
            
            # 2025 Targeted Updates
            'G7 Hiroshima AI Process reporting framework 2025',
            'UN Global Digital Compact AI governance 2025',
            'US-EU Trade and Technology Council AI 2025',
            'China-US AI dialogue 2025 chem-bio weapons',
            'OECD AI Principles 2025 updates',
            'Middle East AI initiatives 2025 Saudi Arabia'
        ]
    }

    
    def __init__(self, config_path="config.yaml"):
        """Initialize scraper with SERP API key from secrets.toml"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load API key from secrets.toml
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from utils import get_api_key
        
        self.api_key = get_api_key('serpapi')
        self.raw_path = Path(self.config['paths']['raw_data'])
        self.raw_path.mkdir(parents=True, exist_ok=True)
    
    def search_policy_events(
        self, 
        query: str, 
        category: str,
        start_date: str = "2022-01-01",
        end_date: str = None,
        num_results: int = 10
    ) -> pd.DataFrame:
        """
        Search for policy events using SERP API Google News
        
        Args:
            query: Search query string
            category: Event category (export_control, regulation, etc.)
            start_date: Start date for search (YYYY-MM-DD)
            end_date: End date for search (YYYY-MM-DD), defaults to today
            num_results: Number of results to retrieve
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Searching: '{query}' ({category})")
        
        # Build SERP API parameters for Google News
        # Note: Google News doesn't support custom date ranges via tbs parameter
        # Using google_news engine instead of google with tbm=nws
        params = {
            'api_key': self.api_key,
            'engine': 'google_news',  # Use dedicated Google News engine
            'q': query,
            'num': num_results,
            'gl': 'us',  # Geolocation
            'hl': 'en'   # Language
        }
        
        try:
            response = requests.get(self.SERP_API_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Debug: Check API response structure
            if 'error' in data:
                logger.error(f"SERP API Error: {data['error']}")
                raise RuntimeError(f"SERP API returned error: {data['error']}")
            
            # Debug: Print available keys
            logger.debug(f"API Response Keys: {list(data.keys())}")
            
            # Parse results
            events = []
            news_results = data.get('news_results', [])
            
            # Debug: Check if results are empty
            if not news_results:
                logger.warning(f"No 'news_results' found. Available keys: {list(data.keys())}")
                logger.warning(f"Full response: {data}")
            
            for result in news_results:
                # Extract source name if it's a dict
                source_name = result.get('source', '')
                if isinstance(source_name, dict):
                    source_name = source_name.get('name', '')
                
                # Use iso_date if available, otherwise regular date
                date_str = result.get('iso_date') or result.get('date', '')
                
                # Try multiple fields for description text
                # Google News API may use 'snippet', 'highlight', or just have title
                snippet = (result.get('snippet') or 
                          result.get('highlight') or 
                          result.get('description') or 
                          '')
                
                events.append({
                    'title': result.get('title', ''),
                    'snippet': snippet,
                    'link': result.get('link', ''),
                    'source': source_name,
                    'date': date_str,
                    'category': category,
                    'query': query
                })
            
            logger.info(f"  Found {len(events)} events")
            return pd.DataFrame(events)
            
        except Exception as e:
            logger.error(f"Failed to search '{query}': {e}")
            raise RuntimeError(
                f"CRITICAL: SERP API request failed for query '{query}'!\n"
                f"Error: {e}\n"
                "Check your API key and quota."
            )
    
    def scrape_all_categories(
        self,
        start_date: str = "2022-01-01",
        end_date: str = None,
        results_per_query: int = 20
    ) -> pd.DataFrame:
        """
        Scrape policy events across all categories
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            results_per_query: Number of results per search query
        """
        print("="*60)
        print("Scraping AI Policy Events with SERP API")
        print("="*60)
        print(f"⚠️  NO SYNTHETIC DATA - Only real news from Google")
        print(f"Date range: {start_date} to {end_date or 'today'}")
        print()
        
        all_events = []
        
        for category, queries in self.SEARCH_QUERIES.items():
            print(f"\nCategory: {category}")
            print("-" * 40)
            
            for query in queries:
                try:
                    df = self.search_policy_events(
                        query=query,
                        category=category,
                        start_date=start_date,
                        end_date=end_date,
                        num_results=results_per_query
                    )
                    all_events.append(df)
                    
                    # Rate limiting - be nice to SERP API
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Skipping query '{query}': {e}")
                    continue
        
        if not all_events:
            raise ValueError("No policy events collected! Check API key and queries.")
        
        # Combine all results
        combined = pd.concat(all_events, ignore_index=True)
        
        # Parse dates (handle both iso_date and date formats)
        combined['parsed_date'] = pd.to_datetime(combined['date'], errors='coerce', utc=True)
        
        # Filter by date range (make sure both are timezone-aware)
        start_dt = pd.to_datetime(start_date, utc=True)
        end_dt = pd.to_datetime(end_date, utc=True) if end_date else pd.Timestamp.now(tz='UTC')
        
        combined = combined[
            (combined['parsed_date'] >= start_dt) & 
            (combined['parsed_date'] <= end_dt)
        ]
        
        # Remove duplicates (same title/source)
        combined = combined.drop_duplicates(subset=['title', 'source'], keep='first')
        
        print(f"\n{'='*60}")
        print(f"Total events collected: {len(combined)}")
        print(f"Date range: {combined['parsed_date'].min()} to {combined['parsed_date'].max()}")
        print(f"Categories: {combined['category'].value_counts().to_dict()}")
        print(f"{'='*60}")
        
        return combined
    
    def extract_structured_events(self, raw_events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert raw scraped news into structured policy events
        Uses keyword matching and date extraction
        """
        print("\nExtracting structured policy events...")
        
        structured_events = []
        
        # Define high-impact keywords for severity scoring
        SEVERITY_KEYWORDS = {
            'ban': 0.9,
            'restriction': 0.8,
            'control': 0.7,
            'limit': 0.7,
            'regulation': 0.6,
            'governance': 0.5,
            'agreement': 0.6,
            'cooperation': 0.4
        }
        
        for _, row in raw_events_df.iterrows():
            title_lower = str(row['title']).lower()
            # Handle NaN snippets - use empty string as fallback
            snippet_lower = str(row['snippet']).lower() if pd.notna(row['snippet']) else ''
            
            # Calculate severity score based on keywords
            severity = 0.5  # Base severity
            for keyword, weight in SEVERITY_KEYWORDS.items():
                if keyword in title_lower or keyword in snippet_lower:
                    severity = max(severity, weight)
            
            # Detect region
            region = 'Global'
            combined_text = title_lower + ' ' + snippet_lower
            if 'china' in combined_text:
                region = 'US-China' if ('us' in combined_text or 'united states' in combined_text) else 'China'
            elif 'eu' in combined_text or 'europe' in combined_text:
                region = 'EU'
            elif 'multilateral' in combined_text or 'international' in combined_text:
                region = 'Multilateral'
            
            # Use snippet if available, otherwise use title as description
            description = row['snippet'] if pd.notna(row['snippet']) and row['snippet'] else row['title']
            
            structured_events.append({
                'date': row['parsed_date'],
                'event_name': row['title'][:100],  # Truncate long titles
                'description': str(description)[:200],
                'category': row['category'],
                'severity': int(severity * 10),  # Scale to 1-10
                'region': region,
                'source': f"{row['source']} - {row['link']}",
                'year_month': row['parsed_date'].strftime('%Y-%m') if pd.notna(row['parsed_date']) else None
            })
        
        structured_df = pd.DataFrame(structured_events)
        
        # Remove events with invalid dates
        structured_df = structured_df.dropna(subset=['date', 'year_month'])
        
        # Sort by date
        structured_df = structured_df.sort_values('date').reset_index(drop=True)
        
        print(f"✓ Extracted {len(structured_df)} structured events")
        
        return structured_df
    
    def save_events(self, raw_df: pd.DataFrame, structured_df: pd.DataFrame):
        """Save both raw and structured event data"""
        
        # Save raw scraped data
        raw_path = self.raw_path / 'policy_events_raw.csv'
        raw_df.to_csv(raw_path, index=False)
        print(f"✓ Saved raw events to {raw_path}")
        
        # Save structured events (for analysis)
        structured_path = Path('data') / 'policy_events.csv'
        structured_df.to_csv(structured_path, index=False)
        print(f"✓ Saved structured events to {structured_path}")
        
        return structured_path
    
    def run(
        self,
        start_date: str = "2022-01-01",
        end_date: str = None,
        results_per_query: int = 20
    ):
        """
        Execute full policy event scraping pipeline
        
        Args:
            start_date: Start date for event search
            end_date: End date for event search
            results_per_query: Number of results per search query
        """
        # Scrape raw events
        raw_df = self.scrape_all_categories(
            start_date=start_date,
            end_date=end_date,
            results_per_query=results_per_query
        )
        
        # Extract structured events
        structured_df = self.extract_structured_events(raw_df)
        
        # Save results
        output_path = self.save_events(raw_df, structured_df)
        
        print(f"\n✓ Policy event scraping complete!")
        print(f"\nSample events:")
        print(structured_df.head(10)[['date', 'event_name', 'category', 'severity']])
        
        return output_path


if __name__ == "__main__":
    scraper = PolicyEventScraper()
    
    # Scrape events from 2022 to present
    # Using fewer results per query to avoid API quota exhaustion
    scraper.run(
        start_date="2022-01-01",
        end_date=None,  # Today
        results_per_query=10  # Reduced from 20 to save API quota
    )
