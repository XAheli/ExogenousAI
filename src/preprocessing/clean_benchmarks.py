"""
Benchmark Data Cleaning
Normalizes and validates benchmark scores
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path


class BenchmarkCleaner:
    """Clean and normalize benchmark data"""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize cleaner with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_path = Path(self.config['paths']['raw_data'])
        self.processed_path = Path(self.config['paths']['processed_data'])
        self.processed_path.mkdir(parents=True, exist_ok=True)
    
    def load_raw_benchmarks(self, filename='pwc_benchmarks.csv'):
        """Load raw benchmark data"""
        filepath = self.raw_path / filename
        print(f"Loading raw benchmarks from {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records")
        
        return df
    
    def clean_data(self, df):
        """Clean and validate benchmark data"""
        print("Cleaning benchmark data...")
        
        initial_count = len(df)
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['date', 'benchmark', 'model_name'])
        print(f"  Removed {initial_count - len(df)} duplicates")
        
        # Validate score ranges (0-100 for most benchmarks)
        df = df[(df['score'] >= 0) & (df['score'] <= 100)]
        
        # Sort by date and benchmark
        df = df.sort_values(['benchmark', 'date', 'rank']).reset_index(drop=True)
        
        # Fill missing ranks within benchmark groups
        df['rank'] = df.groupby(['benchmark', 'date'])['rank'].transform(
            lambda x: x.fillna(method='ffill').fillna(method='bfill')
        )
        
        return df
    
    def normalize_scores(self, df):
        """Normalize scores to comparable scale"""
        print("Normalizing scores...")
        
        # Create normalized score (0-1 scale)
        df['score_normalized'] = df['score'] / 100.0
        
        # Calculate benchmark-specific percentiles
        df['score_percentile'] = df.groupby('benchmark')['score'].transform(
            lambda x: x.rank(pct=True)
        )
        
        return df
    
    def add_derived_metrics(self, df):
        """Add derived metrics for analysis"""
        print("Adding derived metrics...")
        
        # Month-over-month growth rate
        df['score_mom_change'] = df.groupby(['benchmark', 'model_name'])['score'].pct_change()
        
        # Year-over-year growth rate
        df = df.sort_values(['benchmark', 'model_name', 'date'])
        df['score_yoy_change'] = df.groupby(['benchmark', 'model_name'])['score'].pct_change(periods=12)
        
        # Top-1 indicator (best model per benchmark per month)
        df['is_top1'] = df.groupby(['benchmark', 'date'])['score'].transform(
            lambda x: x == x.max()
        ).astype(int)
        
        # Top-3 indicator
        df['is_top3'] = (df['rank'] <= 3).astype(int)
        
        return df
    
    def calculate_aggregate_metrics(self, df):
        """Calculate aggregate benchmark metrics per month"""
        print("Calculating aggregate metrics...")
        
        # Top-1 score per benchmark per month
        top1 = df[df['is_top1'] == 1].groupby(['date', 'benchmark']).agg({
            'score': 'max',
            'model_name': 'first'
        }).reset_index()
        
        top1 = top1.rename(columns={
            'score': 'top1_score',
            'model_name': 'top1_model'
        })
        
        # Average of top-3 scores
        top3 = df[df['is_top3'] == 1].groupby(['date', 'benchmark']).agg({
            'score': 'mean'
        }).reset_index()
        
        top3 = top3.rename(columns={'score': 'top3_avg_score'})
        
        # Merge aggregates
        agg_df = top1.merge(top3, on=['date', 'benchmark'], how='outer')
        
        return agg_df
    
    def validate_data(self, df):
        """Validate data quality"""
        print("\nData Validation:")
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            print("  Warning: Missing values detected:")
            print(missing[missing > 0])
        else:
            print("  ✓ No missing values")
        
        # Check date range
        date_range = (df['date'].max() - df['date'].min()).days
        print(f"  Date range: {date_range} days")
        
        # Check benchmarks coverage
        benchmarks = df['benchmark'].nunique()
        print(f"  Benchmarks covered: {benchmarks}")
        
        # Check data density (records per month)
        df['year_month'] = df['date'].dt.to_period('M')
        density = df.groupby('year_month').size().mean()
        print(f"  Average records per month: {density:.1f}")
        
        return True
    
    def save_cleaned_data(self, df, filename='benchmarks_cleaned.csv'):
        """Save cleaned benchmark data"""
        output_path = self.processed_path / filename
        df.to_csv(output_path, index=False)
        print(f"\nSaved cleaned data to {output_path}")
        
        return output_path
    
    def save_aggregate_data(self, df, filename='benchmarks_aggregate.csv'):
        """Save aggregate benchmark data"""
        output_path = self.processed_path / filename
        df.to_csv(output_path, index=False)
        print(f"Saved aggregate data to {output_path}")
        
        return output_path
    
    def run(self):
        """Execute full cleaning pipeline"""
        print("="*60)
        print("Starting Benchmark Data Cleaning")
        print("="*60)
        
        # Load raw data
        df = self.load_raw_benchmarks()
        
        # Clean data
        df = self.clean_data(df)
        
        # Normalize scores
        df = self.normalize_scores(df)
        
        # Add derived metrics
        df = self.add_derived_metrics(df)
        
        # Validate
        self.validate_data(df)
        
        # Save cleaned data
        cleaned_path = self.save_cleaned_data(df)
        
        # Calculate and save aggregates
        agg_df = self.calculate_aggregate_metrics(df)
        agg_path = self.save_aggregate_data(agg_df)
        
        print(f"\n✓ Benchmark cleaning complete!")
        return cleaned_path, agg_path


if __name__ == "__main__":
    cleaner = BenchmarkCleaner()
    cleaner.run()
