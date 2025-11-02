"""
Event Study Analysis
Calculates abnormal returns around policy events
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import yaml
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class EventStudyAnalyzer:
    """Perform event study analysis on AI capability metrics"""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize analyzer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.processed_path = Path(self.config['paths']['processed_data'])
        self.output_path = Path(self.config['paths']['output'])
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.pre_window = self.config['event_study']['pre_event_window']
        self.post_window = self.config['event_study']['post_event_window']
        self.significance_level = self.config['event_study']['significance_level']
        self.trend_model = self.config['event_study']['trend_model']
    
    def load_merged_data(self):
        """Load merged dataset"""
        print("Loading merged dataset...")
        
        filepath = self.processed_path / 'merged_dataset.csv'
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"  Loaded {len(df)} records")
        return df
    
    def exponential_trend(self, x, a, b):
        """Exponential trend function"""
        return a * np.exp(b * x)
    
    def linear_trend(self, x, a, b):
        """Linear trend function"""
        return a + b * x
    
    def fit_trend(self, dates, values):
        """Fit trend model to pre-event data"""
        # Convert dates to numeric (months since first date)
        x = (dates - dates.min()).dt.days / 30.44
        y = values.values
        
        # Remove NaN values
        valid_mask = ~np.isnan(y)
        x = x[valid_mask]
        y = y[valid_mask]
        
        if len(x) < 3:
            return None, None
        
        try:
            if self.trend_model == 'exponential':
                # Fit exponential trend
                popt, _ = curve_fit(self.exponential_trend, x, y, p0=[y[0], 0.01], maxfev=10000)
            else:
                # Fit linear trend
                popt, _ = curve_fit(self.linear_trend, x, y)
            
            return popt, x.max()
        
        except Exception as e:
            print(f"    Trend fitting failed: {e}")
            return None, None
    
    def calculate_expected_performance(self, dates, trend_params, x_ref):
        """Calculate expected performance based on trend"""
        if trend_params is None:
            return None
        
        # Convert dates to numeric
        x = (dates - dates.min()).dt.days / 30.44 + x_ref
        
        if self.trend_model == 'exponential':
            expected = self.exponential_trend(x, *trend_params)
        else:
            expected = self.linear_trend(x, *trend_params)
        
        return expected
    
    def calculate_abnormal_returns(self, df, event_id, metric='benchmark_composite'):
        """Calculate abnormal returns for a specific event and metric - SKIP IF INSUFFICIENT REAL DATA"""
        # Get event data
        event_data = df[df['event_id'] == event_id].copy()
        
        if len(event_data) == 0:
            return None
        
        # Separate pre, event, and post periods
        pre_data = event_data[event_data['event_period'] == 'pre']
        event_month = event_data[event_data['event_period'] == 'event']
        post_data = event_data[event_data['event_period'] == 'post']
        
        # CRITICAL: Check if we have REAL (non-NaN) data
        pre_real = pre_data[metric].dropna()
        post_real = post_data[metric].dropna()
        
        # Skip if insufficient real data points (need at least 2 for trend)
        if len(pre_real) < 2 or len(post_real) < 1:
            return None
        
        if len(pre_data) == 0 or len(post_data) == 0:
            return None
        
        # Fit trend to pre-event data (only on non-NaN values)
        pre_data_clean = pre_data.dropna(subset=[metric])
        if len(pre_data_clean) < 2:
            return None
            
        trend_params, x_ref = self.fit_trend(pre_data_clean['date'], pre_data_clean[metric])
        
        if trend_params is None:
            return None
        
        # Calculate expected values for post-event period (only where we have real data)
        post_data_clean = post_data.dropna(subset=[metric])
        if len(post_data_clean) == 0:
            return None
            
        post_expected = self.calculate_expected_performance(
            post_data_clean['date'], trend_params, x_ref
        )
        
        if post_expected is None:
            return None
        
        # Calculate abnormal returns (only on real data points)
        post_actual = post_data_clean[metric].values
        post_abnormal = (post_actual - post_expected) / post_expected * 100
        
        # Statistical test (t-test comparing pre vs post abnormal returns)
        pre_data_clean = pre_data.dropna(subset=[metric])
        pre_expected = self.calculate_expected_performance(
            pre_data_clean['date'], trend_params, 0
        )
        pre_abnormal = (pre_data_clean[metric].values - pre_expected) / pre_expected * 100
        
        # T-test
        t_stat, p_value = stats.ttest_ind(post_abnormal, pre_abnormal, nan_policy='omit')
        
        # Compile results
        results = {
            'event_id': event_id,
            'event_name': event_data['event_name'].iloc[0],
            'metric': metric,
            'pre_mean': pre_data[metric].mean(),
            'post_mean': post_data[metric].mean(),
            'pre_abnormal_mean': np.nanmean(pre_abnormal),
            'post_abnormal_mean': np.nanmean(post_abnormal),
            'abnormal_return_change': np.nanmean(post_abnormal),
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < self.significance_level,
            'trend_params': trend_params.tolist()
        }
        
        return results
    
    def analyze_all_events(self, df):
        """Analyze all events for all metrics"""
        print("\nAnalyzing events...")
        
        # Metrics to analyze
        metrics = [
            'benchmark_composite',
            'research_velocity',
            'nvda_return'
        ]
        
        # Get unique events
        events = df[df['event_id'].notna()]['event_id'].unique()
        
        results = []
        
        for event_id in events:
            event_name = df[df['event_id'] == event_id]['event_name'].iloc[0]
            print(f"\n  Event {int(event_id)}: {event_name}")
            
            for metric in metrics:
                if metric not in df.columns or df[metric].isnull().all():
                    continue
                
                result = self.calculate_abnormal_returns(df, event_id, metric)
                
                if result:
                    results.append(result)
                    
                    sig_marker = "***" if result['is_significant'] else ""
                    print(f"    {metric}: AR = {result['abnormal_return_change']:.2f}% "
                          f"(p = {result['p_value']:.3f}) {sig_marker}")
        
        return pd.DataFrame(results)
    
    def calculate_cumulative_abnormal_returns(self, df):
        """Calculate cumulative abnormal returns (CAR) for visualization"""
        print("\nCalculating cumulative abnormal returns...")
        
        # Get event windows
        event_windows = df[df['in_event_window'] == 1].copy()
        
        # For each event, calculate CAR
        car_data = []
        
        for event_id in event_windows['event_id'].unique():
            if pd.isna(event_id):
                continue
            
            event_data = event_windows[event_windows['event_id'] == event_id].copy()
            event_data = event_data.sort_values('date')
            
            # Calculate CAR for benchmark composite
            if 'benchmark_composite_growth' in event_data.columns:
                car = event_data['benchmark_composite_growth'].cumsum()
                
                for idx, row in event_data.iterrows():
                    car_data.append({
                        'date': row['date'],
                        'event_id': event_id,
                        'event_name': row['event_name'],
                        'months_from_event': row['months_from_event'],
                        'CAR': car.loc[idx] if idx in car.index else np.nan
                    })
        
        return pd.DataFrame(car_data)
    
    def generate_event_summary(self, results_df):
        """Generate summary statistics for all events"""
        print("\n" + "="*60)
        print("Event Study Results Summary")
        print("="*60)
        
        # Overall statistics
        significant_count = results_df['is_significant'].sum()
        total_count = len(results_df)
        
        print(f"\nTotal analyses: {total_count}")
        print(f"Significant results (p < {self.significance_level}): {significant_count} "
              f"({significant_count/total_count*100:.1f}%)")
        
        # By metric
        print(f"\nResults by Metric:")
        for metric in results_df['metric'].unique():
            metric_results = results_df[results_df['metric'] == metric]
            sig_count = metric_results['is_significant'].sum()
            avg_ar = metric_results['abnormal_return_change'].mean()
            
            print(f"  {metric}:")
            print(f"    Significant: {sig_count}/{len(metric_results)}")
            print(f"    Avg AR: {avg_ar:.2f}%")
        
        # Most impactful events
        print(f"\nMost Impactful Events (by |AR|):")
        top_events = results_df.nlargest(5, 'abnormal_return_change', keep='all')
        
        for _, row in top_events.iterrows():
            sig = "***" if row['is_significant'] else ""
            print(f"  {row['event_name']} ({row['metric']}): "
                  f"AR = {row['abnormal_return_change']:.2f}% {sig}")
    
    def save_results(self, results_df, car_df):
        """Save event study results"""
        # Save detailed results
        results_path = self.output_path / 'event_study_results.csv'
        results_df.to_csv(results_path, index=False)
        print(f"\nSaved event study results to {results_path}")
        
        # Save CAR data
        car_path = self.output_path / 'cumulative_abnormal_returns.csv'
        car_df.to_csv(car_path, index=False)
        print(f"Saved CAR data to {car_path}")
        
        return results_path, car_path
    
    def run(self):
        """Execute full event study pipeline"""
        print("="*60)
        print("Starting Event Study Analysis")
        print("="*60)
        
        # Load data
        df = self.load_merged_data()
        
        # Analyze all events
        results_df = self.analyze_all_events(df)
        
        # Calculate CAR
        car_df = self.calculate_cumulative_abnormal_returns(df)
        
        # Generate summary
        self.generate_event_summary(results_df)
        
        # Save results
        results_path, car_path = self.save_results(results_df, car_df)
        
        print(f"\n✓ Event study analysis complete!")
        return results_path, car_path


if __name__ == "__main__":
    analyzer = EventStudyAnalyzer()
    analyzer.run()
