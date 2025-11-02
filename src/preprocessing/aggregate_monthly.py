"""
Monthly Aggregation
Aggregates all time series to monthly frequency
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path


class MonthlyAggregator:
    """Aggregate all time series data to monthly frequency"""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize aggregator with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_path = Path(self.config['paths']['raw_data'])
        # EpochAI manual-download path (contains ai_models csvs)
        self.epochai_path = Path(self.config.get('data_sources', {}).get('epochai', {}).get('ai_models_path', "data/raw/epochai/ai_models"))
        self.processed_path = Path(self.config['paths']['processed_data'])
        self.processed_path.mkdir(parents=True, exist_ok=True)
    
    def aggregate_benchmarks(self):
        """Aggregate benchmark data to monthly"""
        print("Aggregating benchmarks to monthly...")
        
        # Load cleaned benchmark data
        filepath = self.processed_path / 'benchmarks_aggregate.csv'
        
        if not filepath.exists():
            print("  Aggregate benchmarks not found, skipping...")
            return None
        
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        
        # Already at monthly frequency from cleaning step
        # Just ensure we have first day of month
        df['date'] = df['date'].dt.to_period('M').dt.to_timestamp()
        
        # Pivot to have benchmarks as columns
        pivot_df = df.pivot_table(
            index='date',
            columns='benchmark',
            values='top1_score',
            aggfunc='mean'
        ).reset_index()
        
        # Flatten column names
        pivot_df.columns = ['date'] + [f'benchmark_{col}' for col in pivot_df.columns[1:]]
        
        print(f"  Aggregated to {len(pivot_df)} monthly records")
        return pivot_df
    
    def aggregate_arxiv(self):
        """Aggregate arXiv submissions to monthly"""
        print("Aggregating arXiv data to monthly...")
        
        filepath = self.raw_path / 'arxiv_submissions.csv'
        
        if not filepath.exists():
            print("  arXiv data not found, skipping...")
            return None
        
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        
        # Ensure monthly aggregation
        df['year_month'] = df['date'].dt.to_period('M')
        
        # Aggregate by month and category
        monthly = df.groupby(['year_month', 'category']).agg({
            'submission_count': 'sum'
        }).reset_index()
        
        # Pivot categories to columns
        pivot_df = monthly.pivot_table(
            index='year_month',
            columns='category',
            values='submission_count',
            aggfunc='sum'
        ).reset_index()
        
        # Flatten column names and rename category columns
        pivot_df.columns.name = None
        category_cols = [col for col in pivot_df.columns if col != 'year_month']
        new_cols = {'year_month': 'year_month'}
        for col in category_cols:
            new_cols[col] = f'arxiv_{col.replace(".", "_")}'
        pivot_df = pivot_df.rename(columns=new_cols)
        
        # Calculate total submissions (only numeric columns)
        arxiv_cols = [col for col in pivot_df.columns if col.startswith('arxiv_')]
        pivot_df['arxiv_total'] = pivot_df[arxiv_cols].sum(axis=1)
        
        # Convert period to timestamp at the end
        pivot_df['date'] = pivot_df['year_month'].dt.to_timestamp()
        pivot_df = pivot_df.drop('year_month', axis=1)
        
        # Reorder columns with date first
        cols = ['date'] + [col for col in pivot_df.columns if col != 'date']
        pivot_df = pivot_df[cols]
        
        print(f"  Aggregated to {len(pivot_df)} monthly records")
        return pivot_df
    
    def aggregate_stocks(self):
        """Aggregate stock data to monthly"""
        print("Aggregating stock data to monthly...")
        
        filepath = self.raw_path / 'nvidia_stock.csv'
        
        if not filepath.exists():
            print("  Stock data not found, skipping...")
            return None
        
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        
        # Convert to start of month for consistency with other data
        df['date'] = df['date'].dt.to_period('M').dt.to_timestamp()
        
        # Data already has correct column names from Alpha Vantage scraper
        # Just select what we need
        df = df[['date', 'nvda_price', 'nvda_volume']].copy()
        
        # Calculate month-over-month return
        df['nvda_return'] = df['nvda_price'].pct_change()
        
        print(f"  Aggregated to {len(df)} monthly records")
        return df

    def aggregate_epochai(self):
        """Aggregate EpochAI model training compute to monthly"""
        print("Aggregating EpochAI model compute to monthly...")

        if not self.epochai_path.exists():
            print(f"  EpochAI path {self.epochai_path} not found, skipping...")
            return None

        # Read all CSV files in the directory
        all_files = list(self.epochai_path.glob('*.csv'))

        if not all_files:
            print(f"  No CSV files found in {self.epochai_path}, skipping...")
            return None

        parts = []
        for fp in all_files:
            try:
                part = pd.read_csv(fp)
                part['source_file'] = fp.name
                parts.append(part)
            except Exception as e:
                print(f"  Warning: failed to read {fp}: {e}")

        if not parts:
            print("  No readable EpochAI CSVs, skipping...")
            return None

        df = pd.concat(parts, ignore_index=True, sort=False)

        # Robust column detection
        col_map = {c.lower(): c for c in df.columns}
        # Publication date
        date_col = None
        for candidate in ['publication date', 'publication_date', 'date', 'release date', 'release_date']:
            if candidate in col_map:
                date_col = col_map[candidate]
                break

        # Training compute
        compute_col = None
        for candidate in ['training compute (flop)', 'training_compute_(flop)', 'training compute', 'training_compute', 'training compute (flops)']:
            if candidate in col_map:
                compute_col = col_map[candidate]
                break

        if date_col is None or compute_col is None:
            print("  EpochAI CSVs do not contain expected 'Publication date' and 'Training compute (FLOP)' columns.\n"
                  "  Please ensure the EpochAI CSVs include these columns.\n"
                  "  Found columns: " + ",".join(list(df.columns)))
            raise ValueError("EpochAI CSV schema mismatch - required columns missing")

        # Parse dates and numeric compute
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df['training_compute_flop'] = pd.to_numeric(df[compute_col], errors='coerce')

        df = df.dropna(subset=[date_col])

        # Aggregate by month: sum of training compute released that month
        df['year_month'] = df[date_col].dt.to_period('M')
        monthly = df.groupby('year_month').agg({'training_compute_flop': 'sum'}).reset_index()
        monthly['date'] = monthly['year_month'].dt.to_timestamp()
        monthly = monthly[['date', 'training_compute_flop']]
        monthly = monthly.rename(columns={'training_compute_flop': 'epochai_training_compute'})

        print(f"  Aggregated EpochAI compute to {len(monthly)} monthly records")
        return monthly
    
    def merge_all_monthly_data(self, benchmarks_df, arxiv_df, stocks_df):
        """Merge all monthly time series"""
        print("Merging all monthly data...")
        
        # Start with date range
        all_dates = []
        
        # Also include EpochAI compute series
        epochai_df = self.aggregate_epochai()

        for df in [benchmarks_df, arxiv_df, stocks_df, epochai_df]:
            if df is not None:
                all_dates.extend(df['date'].tolist())
        
        if not all_dates:
            raise ValueError("No data available to merge")
        
        # Create complete date range
        min_date = min(all_dates)
        max_date = max(all_dates)
        
        date_range = pd.date_range(start=min_date, end=max_date, freq='MS')
        merged_df = pd.DataFrame({'date': date_range})
        
        # Merge each dataset
        if benchmarks_df is not None:
            merged_df = merged_df.merge(benchmarks_df, on='date', how='left')
        
        if arxiv_df is not None:
            merged_df = merged_df.merge(arxiv_df, on='date', how='left')
        
        if stocks_df is not None:
            merged_df = merged_df.merge(stocks_df, on='date', how='left')
        if epochai_df is not None:
            merged_df = merged_df.merge(epochai_df, on='date', how='left')
        # Sort by date
        merged_df = merged_df.sort_values('date').reset_index(drop=True)
        
        print(f"  Merged to {len(merged_df)} monthly records with {len(merged_df.columns)} columns")
        
        return merged_df
    
    def add_time_features(self, df):
        """Add time-based features"""
        print("Adding time features...")
        
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        # Months since start
        df['months_elapsed'] = (df['date'] - df['date'].min()).dt.days / 30.44
        
        return df
    
    def save_monthly_data(self, df, filename='monthly_aggregated.csv'):
        """Save monthly aggregated data"""
        output_path = self.processed_path / filename
        df.to_csv(output_path, index=False)
        print(f"\nSaved monthly data to {output_path}")
        
        # Summary statistics
        print(f"\nMonthly Data Summary:")
        print(f"  Records: {len(df)}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Missing data: {df.isnull().sum().sum()} cells")
        
        return output_path
    
    def run(self):
        """Execute full aggregation pipeline"""
        print("="*60)
        print("Starting Monthly Aggregation")
        print("="*60)
        
        # Aggregate each data source
        benchmarks_df = self.aggregate_benchmarks()
        arxiv_df = self.aggregate_arxiv()
        stocks_df = self.aggregate_stocks()
        
        # Merge all data
        merged_df = self.merge_all_monthly_data(benchmarks_df, arxiv_df, stocks_df)
        
        # Add time features
        merged_df = self.add_time_features(merged_df)
        
        # Save
        output_path = self.save_monthly_data(merged_df)
        
        print(f"\n✓ Monthly aggregation complete!")
        return output_path


if __name__ == "__main__":
    aggregator = MonthlyAggregator()
    aggregator.run()
