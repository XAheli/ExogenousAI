"""
Meta-Forecasting Analysis
Compares and decomposes uncertainty across different forecasts
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from scipy import stats


class MetaForecastingAnalyzer:
    """Analyze and compare multiple AI timeline forecasts"""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize analyzer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_path = Path(self.config['paths']['output'])
        self.baselines = self.config['meta_forecasting']['baselines']
    
    def load_exogenous_forecasts(self):
        """Load ExogenousAI timeline estimates"""
        print("Loading ExogenousAI forecasts...")
        
        filepath = self.output_path / 'agi_timeline_estimates.csv'
        df = pd.read_csv(filepath)
        
        # Reformat for comparison
        exogenous_forecasts = []
        
        for _, row in df.iterrows():
            exogenous_forecasts.append({
                'source': f"ExogenousAI - {row['scenario']}",
                'model': 'Policy-Adjusted',
                'median_year': row['median_year'],
                'p10_year': row['p10_year'],
                'p90_year': row['p90_year'],
                'method': 'Event Study + Monte Carlo'
            })
        
        return pd.DataFrame(exogenous_forecasts)
    
    def load_baseline_forecasts(self):
        """Load baseline forecasts from literature"""
        print("Loading baseline forecasts...")
        
        baseline_forecasts = []
        
        for name, info in self.baselines.items():
            baseline_forecasts.append({
                'source': name.upper().replace('_', ' '),
                'model': info['source'],
                'median_year': info['median_year'],
                'p10_year': info.get('p10_year', np.nan),
                'p90_year': info.get('p90_year', np.nan),
                'method': 'Literature Baseline'
            })
        
        return pd.DataFrame(baseline_forecasts)
    
    def calculate_forecast_variance(self, all_forecasts):
        """Calculate variance across all forecasts"""
        print("\nCalculating forecast variance...")
        
        # Overall variance
        median_variance = all_forecasts['median_year'].var()
        median_std = all_forecasts['median_year'].std()
        median_range = all_forecasts['median_year'].max() - all_forecasts['median_year'].min()
        
        # By method
        method_variance = all_forecasts.groupby('method')['median_year'].agg(['mean', 'std', 'var'])
        
        variance_results = {
            'total_variance': median_variance,
            'total_std': median_std,
            'total_range': median_range,
            'by_method': method_variance
        }
        
        return variance_results
    
    def decompose_uncertainty(self, exogenous_forecasts):
        """Decompose uncertainty into technical, economic, and policy components"""
        print("\nDecomposing uncertainty...")
        
        # Calculate variance of ExogenousAI scenarios (policy uncertainty)
        policy_variance = exogenous_forecasts['median_year'].var()
        
        # Estimate technical uncertainty from confidence intervals
        exogenous_forecasts['ci_width'] = exogenous_forecasts['p90_year'] - exogenous_forecasts['p10_year']
        technical_variance = exogenous_forecasts['ci_width'].mean() ** 2 / 16  # Approximation
        
        # Economic uncertainty (residual from stock volatility proxy)
        # Simplified: assume 30% of technical variance
        economic_variance = technical_variance * 0.3
        
        # Total variance
        total_variance = policy_variance + technical_variance + economic_variance
        
        # Proportions
        decomposition = {
            'technical_variance': technical_variance,
            'economic_variance': economic_variance,
            'policy_variance': policy_variance,
            'total_variance': total_variance,
            'technical_proportion': technical_variance / total_variance,
            'economic_proportion': economic_variance / total_variance,
            'policy_proportion': policy_variance / total_variance
        }
        
        return decomposition
    
    def analyze_policy_assumptions(self, all_forecasts, exogenous_forecasts):
        """Analyze implicit policy assumptions in baseline forecasts"""
        print("\nAnalyzing policy assumptions...")
        
        # Compare baseline forecasts to ExogenousAI scenarios
        baseline_df = all_forecasts[all_forecasts['method'] == 'Literature Baseline']
        
        analysis = []
        
        for _, baseline in baseline_df.iterrows():
            # Find closest ExogenousAI scenario
            exog_median = exogenous_forecasts['median_year'].values
            baseline_year = baseline['median_year']
            
            closest_idx = np.argmin(np.abs(exog_median - baseline_year))
            closest_scenario = exogenous_forecasts.iloc[closest_idx]
            
            analysis.append({
                'baseline_source': baseline['source'],
                'closest_scenario': closest_scenario['source'],
                'year_difference': baseline_year - closest_scenario['median_year'],
                'implied_policy_stance': self._classify_policy_stance(
                    baseline_year, exogenous_forecasts
                )
            })
        
        return pd.DataFrame(analysis)
    
    def _classify_policy_stance(self, year, exogenous_forecasts):
        """Classify implicit policy stance based on timeline"""
        scenarios = exogenous_forecasts.set_index('source')['median_year'].to_dict()
        
        # Find which scenario is closest
        baseline_key = list(self.config['scenarios'].keys())[0]  # 'baseline'
        
        if 'Status Quo' in scenarios:
            baseline_year = scenarios['Status Quo']
            
            if year < baseline_year - 2:
                return 'Optimistic (Low Policy Risk)'
            elif year > baseline_year + 2:
                return 'Pessimistic (High Policy Risk)'
            else:
                return 'Neutral (Status Quo)'
        
        return 'Unknown'
    
    def calculate_policy_contribution(self, all_forecasts):
        """Calculate how much policy assumptions contribute to forecast variance"""
        print("\nCalculating policy contribution to variance...")
        
        # Total variance across all forecasts
        total_var = all_forecasts['median_year'].var()
        
        # Variance within ExogenousAI scenarios (policy-driven)
        exogenous_df = all_forecasts[all_forecasts['method'] == 'Event Study + Monte Carlo']
        policy_var_median = exogenous_df['median_year'].var()
        
        # Also calculate variance in mean years and CI widths
        policy_var_mean = exogenous_df['p90_year'].var() if 'p90_year' in exogenous_df.columns else 0
        
        # Use mean years for policy variance if median variance is 0
        if policy_var_median == 0 and 'mean_year' in exogenous_df.columns:
            # Calculate from mean years instead
            mean_years = []
            timeline_df = pd.read_csv(self.output_path / 'agi_timeline_estimates.csv')
            policy_var_median = timeline_df['mean_months'].var() / 144  # Convert to years²
        
        policy_var = policy_var_median
        
        # Variance within baseline forecasts (implicit assumptions)
        baseline_df = all_forecasts[all_forecasts['method'] == 'Literature Baseline']
        baseline_var = baseline_df['median_year'].var()
        
        # Policy contribution - if median variance is 0, use a minimum based on CI spread
        if policy_var == 0:
            # Calculate effective policy variance from confidence interval differences
            if len(exogenous_df) > 0:
                ci_widths = exogenous_df['p90_year'] - exogenous_df['p10_year']
                policy_var = ci_widths.var() / 16  # Approximate variance from CI range variance
        
        policy_contribution = policy_var / total_var if total_var > 0 else 0
        
        contribution_results = {
            'total_variance': total_var,
            'policy_variance': policy_var,
            'baseline_variance': baseline_var,
            'policy_contribution_pct': policy_contribution * 100,
            'note': 'Policy variance from CI spread' if policy_var_median == 0 else 'Policy variance from median years'
        }
        
        return contribution_results
    
    def save_results(self, all_forecasts, variance_results, decomposition, 
                    policy_analysis, contribution):
        """Save meta-forecasting results"""
        print("\nSaving results...")
        
        # Save all forecasts comparison
        forecasts_path = self.output_path / 'forecast_comparison.csv'
        all_forecasts.to_csv(forecasts_path, index=False)
        print(f"  Saved forecast comparison to {forecasts_path}")
        
        # Save policy analysis
        policy_path = self.output_path / 'policy_assumptions_analysis.csv'
        policy_analysis.to_csv(policy_path, index=False)
        print(f"  Saved policy analysis to {policy_path}")
        
        # Save decomposition
        decomp_path = self.output_path / 'uncertainty_decomposition.csv'
        pd.DataFrame([decomposition]).to_csv(decomp_path, index=False)
        print(f"  Saved uncertainty decomposition to {decomp_path}")
        
        # Save contribution analysis
        contrib_path = self.output_path / 'policy_contribution.csv'
        pd.DataFrame([contribution]).to_csv(contrib_path, index=False)
        print(f"  Saved policy contribution to {contrib_path}")
        
        return forecasts_path
    
    def generate_summary(self, all_forecasts, variance_results, decomposition, contribution):
        """Generate summary of meta-forecasting analysis"""
        print("\n" + "="*60)
        print("Meta-Forecasting Analysis Results")
        print("="*60)
        
        print(f"\nForecast Comparison:")
        print(f"  Total forecasts analyzed: {len(all_forecasts)}")
        print(f"  Timeline range: {int(all_forecasts['median_year'].min())} - "
              f"{int(all_forecasts['median_year'].max())}")
        print(f"  Overall variance: {variance_results['total_variance']:.2f} years²")
        print(f"  Overall std dev: {variance_results['total_std']:.2f} years")
        
        print(f"\nUncertainty Decomposition:")
        print(f"  Technical uncertainty: {decomposition['technical_proportion']*100:.1f}%")
        print(f"  Economic uncertainty: {decomposition['economic_proportion']*100:.1f}%")
        print(f"  Policy uncertainty: {decomposition['policy_proportion']*100:.1f}%")
        
        print(f"\nPolicy Contribution:")
        print(f"  Policy assumptions explain {contribution['policy_contribution_pct']:.1f}% "
              f"of forecast variance")
        
        print(f"\nIndividual Forecasts:")
        for _, row in all_forecasts.iterrows():
            ci_str = ""
            if not pd.isna(row['p10_year']) and not pd.isna(row['p90_year']):
                ci_str = f" [{int(row['p10_year'])}-{int(row['p90_year'])}]"
            print(f"  {row['source']}: {int(row['median_year'])}{ci_str}")
    
    def run(self):
        """Execute full meta-forecasting analysis"""
        print("="*60)
        print("Starting Meta-Forecasting Analysis")
        print("="*60)
        
        # Load forecasts
        exogenous_forecasts = self.load_exogenous_forecasts()
        baseline_forecasts = self.load_baseline_forecasts()
        
        # Combine all forecasts
        all_forecasts = pd.concat([exogenous_forecasts, baseline_forecasts], ignore_index=True)
        
        # Calculate variance
        variance_results = self.calculate_forecast_variance(all_forecasts)
        
        # Decompose uncertainty
        decomposition = self.decompose_uncertainty(exogenous_forecasts)
        
        # Analyze policy assumptions
        policy_analysis = self.analyze_policy_assumptions(all_forecasts, exogenous_forecasts)
        
        # Calculate policy contribution
        contribution = self.calculate_policy_contribution(all_forecasts)
        
        # Save results
        forecasts_path = self.save_results(
            all_forecasts, variance_results, decomposition, 
            policy_analysis, contribution
        )
        
        # Generate summary
        self.generate_summary(all_forecasts, variance_results, decomposition, contribution)
        
        print(f"\n✓ Meta-forecasting analysis complete!")
        return forecasts_path


if __name__ == "__main__":
    analyzer = MetaForecastingAnalyzer()
    analyzer.run()
