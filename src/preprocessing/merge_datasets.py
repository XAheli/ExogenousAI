"""
Dataset Merger
Merges all processed datasets with policy events
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path


class DatasetMerger:
    """Merge all datasets for analysis"""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize merger with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.processed_path = Path(self.config['paths']['processed_data'])
        self.policy_events_path = Path(self.config['paths']['policy_events'])
    
    def load_monthly_data(self):
        """Load monthly aggregated data"""
        print("Loading monthly aggregated data...")
        
        filepath = self.processed_path / 'monthly_aggregated.csv'
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"  Loaded {len(df)} monthly records")
        return df
    
    def load_policy_events(self):
        """Load policy events timeline"""
        print("Loading policy events...")
        
        df = pd.read_csv(self.policy_events_path)
        df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)  # Remove timezone
        df['pre_window_start'] = pd.to_datetime(df['pre_window_start'], utc=True).dt.tz_localize(None)
        df['post_window_end'] = pd.to_datetime(df['post_window_end'], utc=True).dt.tz_localize(None)
        
        print(f"  Loaded {len(df)} policy events")
        return df
    
    def mark_event_periods(self, monthly_df, events_df):
        """Mark event periods in monthly data"""
        print("Marking event periods...")
        
        # Initialize event markers
        monthly_df['in_event_window'] = 0
        monthly_df['event_id'] = None
        monthly_df['event_name'] = None
        monthly_df['event_type'] = None
        monthly_df['event_period'] = None  # 'pre', 'event', 'post', or None
        monthly_df['months_from_event'] = None
        
        # Mark each event window
        for _, event in events_df.iterrows():
            event_date = event['date']
            pre_start = event['pre_window_start']
            post_end = event['post_window_end']
            
            # Mark pre-event period
            pre_mask = (monthly_df['date'] >= pre_start) & (monthly_df['date'] < event_date)
            monthly_df.loc[pre_mask, 'in_event_window'] = 1
            monthly_df.loc[pre_mask, 'event_id'] = event['event_id']
            monthly_df.loc[pre_mask, 'event_name'] = event['event_name']
            monthly_df.loc[pre_mask, 'event_type'] = event['event_type']
            monthly_df.loc[pre_mask, 'event_period'] = 'pre'
            
            # Mark event month
            event_mask = (monthly_df['date'] == event_date)
            monthly_df.loc[event_mask, 'in_event_window'] = 1
            monthly_df.loc[event_mask, 'event_id'] = event['event_id']
            monthly_df.loc[event_mask, 'event_name'] = event['event_name']
            monthly_df.loc[event_mask, 'event_type'] = event['event_type']
            monthly_df.loc[event_mask, 'event_period'] = 'event'
            
            # Mark post-event period
            post_mask = (monthly_df['date'] > event_date) & (monthly_df['date'] <= post_end)
            monthly_df.loc[post_mask, 'in_event_window'] = 1
            monthly_df.loc[post_mask, 'event_id'] = event['event_id']
            monthly_df.loc[post_mask, 'event_name'] = event['event_name']
            monthly_df.loc[post_mask, 'event_type'] = event['event_type']
            monthly_df.loc[post_mask, 'event_period'] = 'post'
            
            # Calculate months from event
            all_event_mask = (monthly_df['date'] >= pre_start) & (monthly_df['date'] <= post_end)
            monthly_df.loc[all_event_mask, 'months_from_event'] = \
                ((monthly_df.loc[all_event_mask, 'date'] - event_date).dt.days / 30.44).round()
        
        print(f"  Marked {monthly_df['in_event_window'].sum()} months in event windows")
        return monthly_df
    
    def add_event_severity(self, monthly_df, events_df):
        """Add event severity scores"""
        print("Adding event severity...")
        
        # Create severity mapping
        severity_map = dict(zip(events_df['event_id'], events_df['severity']))
        
        # Add severity column
        monthly_df['event_severity'] = monthly_df['event_id'].map(severity_map)
        
        return monthly_df
    
    def calculate_composite_metrics(self, df):
        """Calculate composite AI capability metrics"""
        print("Calculating composite metrics...")
        
        # Benchmark composite (average of available benchmarks)
        benchmark_cols = [col for col in df.columns if col.startswith('benchmark_')]
        
        if benchmark_cols:
            df['benchmark_composite'] = df[benchmark_cols].mean(axis=1, skipna=True)
        
        # Research velocity composite
        arxiv_cols = [col for col in df.columns if col.startswith('arxiv_') and col != 'arxiv_total']
        
        if arxiv_cols:
            df['research_velocity'] = df[arxiv_cols].sum(axis=1, skipna=True)
        
        # Investment sentiment (normalized stock returns)
        if 'nvda_return' in df.columns:
            df['investment_sentiment'] = df['nvda_return'] / 100.0  # Convert to decimal
        
        return df
    
    def calculate_growth_rates(self, df):
        """Calculate growth rates AFTER interpolation"""
        print("Calculating growth rates...")
        
        # Calculate percentage changes for metrics
        if 'benchmark_composite' in df.columns:
            df['benchmark_composite_growth'] = df['benchmark_composite'].pct_change() * 100  # Convert to percentage
        
        if 'research_velocity' in df.columns:
            df['research_velocity_growth'] = df['research_velocity'].pct_change() * 100  # Convert to percentage
        
        return df
    
    def interpolate_missing_values(self, df):
        """NO INTERPOLATION - Keep only real data points"""
        print("Skipping interpolation - using only real data points...")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        missing_count = df[numeric_cols].isnull().sum().sum()
        print(f"  Missing values preserved: {missing_count} (real data only)")
        
        return df
    
    def filter_valid_date_range(self, df):
        """Filter to date range where we have meaningful modern AI data (2020+)"""
        print("Filtering to valid date range (2020 onwards for modern AI era)...")
        
        # Use 2020+ to capture modern AI development with 2 years baseline before our collected data
        df = df[df['date'] >= '2020-01-01'].copy()
        
        print(f"  Filtered to {len(df)} records (2020-2025)")
        return df
    
    def save_merged_data(self, df, filename='merged_dataset.csv'):
        """Save merged dataset"""
        output_path = self.processed_path / filename
        df.to_csv(output_path, index=False)
        print(f"\nSaved merged dataset to {output_path}")
        
        # Summary statistics
        print(f"\nMerged Dataset Summary:")
        print(f"  Records: {len(df)}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Event windows: {df['in_event_window'].sum()} months")
        print(f"  Non-event periods: {(df['in_event_window'] == 0).sum()} months")
        
        return output_path
    
    def run(self):
        """Execute full merge pipeline"""
        print("="*60)
        print("Starting Dataset Merge")
        print("="*60)
        
        # Load data
        monthly_df = self.load_monthly_data()
        events_df = self.load_policy_events()
        
        # Filter to valid date range (2022+)
        monthly_df = self.filter_valid_date_range(monthly_df)
        
        # Mark event periods
        monthly_df = self.mark_event_periods(monthly_df, events_df)
        
        # Add event severity
        monthly_df = self.add_event_severity(monthly_df, events_df)
        
        # Calculate composite metrics
        monthly_df = self.calculate_composite_metrics(monthly_df)
        
        # Interpolate missing values
        monthly_df = self.interpolate_missing_values(monthly_df)
        
        # Calculate growth rates AFTER interpolation
        monthly_df = self.calculate_growth_rates(monthly_df)
        
        # Save
        output_path = self.save_merged_data(monthly_df)
        
        print(f"\n✓ Dataset merge complete!")
        return output_path


if __name__ == "__main__":
    merger = DatasetMerger()
    merger.run()
