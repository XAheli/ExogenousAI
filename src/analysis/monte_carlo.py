"""
Monte Carlo Scenario Modeling
Simulates AI capability trajectories under different policy scenarios
"""

import pandas as pd
import numpy as np
from scipy import stats
import yaml
from pathlib import Path
from tqdm import tqdm


class MonteCarloSimulator:
    """Monte Carlo simulation for AI capability scenarios"""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize simulator with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.processed_path = Path(self.config['paths']['processed_data'])
        self.output_path = Path(self.config['paths']['output'])
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.n_iterations = self.config['monte_carlo']['n_iterations']
        self.forecast_horizon = self.config['monte_carlo']['forecast_horizon']
        self.random_seed = self.config['monte_carlo']['random_seed']
        self.confidence_intervals = self.config['monte_carlo']['confidence_intervals']
        
        self.scenarios = self.config['scenarios']
        
        np.random.seed(self.random_seed)
    
    def load_data(self):
        """Load historical data for baseline"""
        print("Loading historical data...")
        
        filepath = self.processed_path / 'merged_dataset.csv'
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"  Loaded {len(df)} historical records")
        return df
    
    def fit_baseline_trend(self, df):
        """Fit baseline exponential trend to historical data - ONLY REAL DATA POINTS"""
        print("Fitting baseline trend to REAL data points only...")
        
        # Use benchmark composite as primary metric
        data = df[['date', 'benchmark_composite']].dropna()
        
        if len(data) == 0:
            raise ValueError("No valid benchmark data - cannot fit baseline trend")
        
        if len(data) < 3:
            raise ValueError(f"Insufficient real data points ({len(data)}) - need at least 3 for trend fitting")
        
        print(f"  Using {len(data)} real benchmark data points (no interpolation)")
        
        # Convert to months elapsed
        data['months'] = (data['date'] - data['date'].min()).dt.days / 30.44
        
        # Fit exponential trend: y = a * exp(b * x)
        x = data['months'].values
        y = data['benchmark_composite'].values
        
        # Log-transform for linear regression
        log_y = np.log(y)
        
        # Linear regression on log-transformed data
        coeffs = np.polyfit(x, log_y, 1)
        
        # Extract parameters
        growth_rate = coeffs[0]  # Monthly growth rate
        initial_value = np.exp(coeffs[1])
        
        # Calculate historical volatility
        fitted = initial_value * np.exp(growth_rate * x)
        residuals = y - fitted
        volatility = np.std(residuals / fitted)
        
        baseline_params = {
            'initial_value': initial_value,
            'growth_rate': growth_rate,
            'volatility': volatility,
            'last_actual_value': y[-1],
            'last_date': data['date'].max()
        }
        
        print(f"  Growth rate: {growth_rate*100:.3f}% per month ({growth_rate*12*100:.1f}% per year)")
        print(f"  Volatility: {volatility*100:.2f}%")
        
        return baseline_params
    
    def simulate_trajectory(self, baseline_params, scenario_config, horizon):
        """Simulate single trajectory"""
        # Extract parameters
        initial = baseline_params['last_actual_value']
        growth_rate = baseline_params['growth_rate']
        hist_vol = baseline_params['volatility']
        
        # Policy adjustments
        policy_factor = scenario_config['policy_factor']
        policy_sigma = scenario_config['policy_sigma']
        
        # Adjust growth rate for policy
        adjusted_growth = growth_rate * policy_factor
        
        # Combined volatility
        total_sigma = np.sqrt(hist_vol**2 + policy_sigma**2)
        
        # Simulate monthly values
        trajectory = [initial]
        
        for month in range(1, horizon + 1):
            # Geometric Brownian Motion
            shock = np.random.normal(0, total_sigma)
            next_value = trajectory[-1] * np.exp(adjusted_growth + shock)
            
            # Cap at 100 (perfect score)
            next_value = min(next_value, 100.0)
            
            trajectory.append(next_value)
        
        return np.array(trajectory)
    
    def run_scenario_simulations(self, baseline_params):
        """Run Monte Carlo simulations for all scenarios"""
        print("\nRunning Monte Carlo simulations...")
        
        results = {}
        
        for scenario_name, scenario_config in self.scenarios.items():
            print(f"\n  Scenario: {scenario_config['name']}")
            
            trajectories = []
            
            for i in tqdm(range(self.n_iterations), desc=f"  Simulating"):
                trajectory = self.simulate_trajectory(
                    baseline_params,
                    scenario_config,
                    self.forecast_horizon
                )
                trajectories.append(trajectory)
            
            trajectories = np.array(trajectories)
            
            # Calculate statistics
            results[scenario_name] = {
                'trajectories': trajectories,
                'config': scenario_config,
                'statistics': self.calculate_statistics(trajectories)
            }
        
        return results
    
    def calculate_statistics(self, trajectories):
        """Calculate summary statistics for trajectories"""
        stats_dict = {
            'mean': np.mean(trajectories, axis=0),
            'median': np.median(trajectories, axis=0),
            'std': np.std(trajectories, axis=0)
        }
        
        # Calculate percentiles
        for ci in self.confidence_intervals:
            percentile = ci * 100
            stats_dict[f'p{int(percentile)}'] = np.percentile(trajectories, percentile, axis=0)
        
        return stats_dict
    
    def estimate_agi_timeline(self, results, agi_threshold=95.0):
        """Estimate time to AGI (threshold crossing)"""
        print("\nEstimating AGI timelines...")
        print(f"  Using AGI threshold: {agi_threshold} (adjusted for benchmark saturation)")
        
        timeline_results = []
        
        for scenario_name, result in results.items():
            trajectories = result['trajectories']
            
            # Find first month crossing threshold for each trajectory
            crossing_months = []
            
            for trajectory in trajectories:
                crossing_idx = np.where(trajectory >= agi_threshold)[0]
                
                if len(crossing_idx) > 0:
                    crossing_months.append(crossing_idx[0])
                else:
                    crossing_months.append(self.forecast_horizon)  # Beyond horizon
            
            crossing_months = np.array(crossing_months)
            
            # Calculate statistics
            timeline_results.append({
                'scenario': result['config']['name'],
                'median_months': np.median(crossing_months),
                'mean_months': np.mean(crossing_months),
                'p10_months': np.percentile(crossing_months, 10),
                'p90_months': np.percentile(crossing_months, 90),
                'probability_within_horizon': np.mean(crossing_months < self.forecast_horizon),
                'threshold': agi_threshold
            })
        
        timeline_df = pd.DataFrame(timeline_results)
        
        # Convert to years and dates
        baseline_date = pd.to_datetime('2025-11-01')  # Current date
        
        for col in ['median_months', 'mean_months', 'p10_months', 'p90_months']:
            timeline_df[col.replace('_months', '_year')] = \
                (baseline_date + pd.to_timedelta(timeline_df[col] * 30.44, unit='D')).dt.year
        
        return timeline_df
    
    def save_results(self, results, timeline_df, baseline_params):
        """Save simulation results"""
        print("\nSaving results...")
        
        # Save timeline estimates
        timeline_path = self.output_path / 'agi_timeline_estimates.csv'
        timeline_df.to_csv(timeline_path, index=False)
        print(f"  Saved timeline estimates to {timeline_path}")
        
        # Save summary statistics for each scenario
        for scenario_name, result in results.items():
            stats = result['statistics']
            
            # Create dataframe with all statistics
            months = np.arange(0, self.forecast_horizon + 1)
            
            stats_df = pd.DataFrame({
                'month': months,
                'mean': stats['mean'],
                'median': stats['median'],
                'std': stats['std']
            })
            
            # Add percentiles
            for ci in self.confidence_intervals:
                percentile = int(ci * 100)
                stats_df[f'p{percentile}'] = stats[f'p{percentile}']
            
            # Save
            stats_path = self.output_path / f'scenario_{scenario_name}_statistics.csv'
            stats_df.to_csv(stats_path, index=False)
        
        # Save baseline parameters
        baseline_path = self.output_path / 'baseline_parameters.csv'
        pd.DataFrame([baseline_params]).to_csv(baseline_path, index=False)
        
        print(f"  Saved scenario statistics")
        
        return timeline_path
    
    def generate_summary(self, timeline_df):
        """Generate summary of results"""
        print("\n" + "="*60)
        print("Monte Carlo Simulation Results")
        print("="*60)
        
        print(f"\nAGI Timeline Estimates (Benchmark Score ≥ {int(timeline_df['threshold'].iloc[0])}):")
        print(f"{'Scenario':<30} {'Median Year':<15} {'10th-90th %ile':<20}")
        print("-" * 65)
        
        for _, row in timeline_df.iterrows():
            percentile_range = f"{int(row['p10_year'])}-{int(row['p90_year'])}"
            print(f"{row['scenario']:<30} {int(row['median_year']):<15} {percentile_range:<20}")
        
        print(f"\nTimeline Spread:")
        min_year = timeline_df['median_year'].min()
        max_year = timeline_df['median_year'].max()
        print(f"  Range: {int(max_year - min_year)} years ({int(min_year)} to {int(max_year)})")
        print(f"  Median variance: {timeline_df['median_year'].std():.1f} years")
    
    def run(self):
        """Execute full Monte Carlo simulation pipeline"""
        print("="*60)
        print("Starting Monte Carlo Scenario Modeling")
        print("="*60)
        
        # Load data
        df = self.load_data()
        
        # Fit baseline
        baseline_params = self.fit_baseline_trend(df)
        
        # Run simulations
        results = self.run_scenario_simulations(baseline_params)
        
        # Estimate AGI timelines
        timeline_df = self.estimate_agi_timeline(results)
        
        # Save results
        timeline_path = self.save_results(results, timeline_df, baseline_params)
        
        # Generate summary
        self.generate_summary(timeline_df)
        
        print(f"\n✓ Monte Carlo simulation complete!")
        return timeline_path, results


if __name__ == "__main__":
    simulator = MonteCarloSimulator()
    simulator.run()
