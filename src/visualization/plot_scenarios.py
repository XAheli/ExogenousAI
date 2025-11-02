"""
Scenario Modeling Visualization
Creates plots for Monte Carlo scenario analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path


class ScenarioPlotter:
    """Create visualizations for scenario modeling"""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize plotter with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_path = Path(self.config['paths']['output'])
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use(self.config['visualization']['style'])
        sns.set_palette(self.config['visualization']['color_palette'])
        
        self.figsize = (
            self.config['visualization']['figure_size']['width'],
            self.config['visualization']['figure_size']['height']
        )
        self.dpi = self.config['visualization']['dpi']
        
        self.scenarios = self.config['scenarios']
    
    def load_data(self):
        """Load scenario modeling results"""
        print("Loading scenario data...")
        
        # Timeline estimates
        timeline_path = self.output_path / 'agi_timeline_estimates.csv'
        timeline_df = pd.read_csv(timeline_path)
        
        # Scenario statistics
        scenario_stats = {}
        
        for scenario_name in self.scenarios.keys():
            stats_path = self.output_path / f'scenario_{scenario_name}_statistics.csv'
            
            if stats_path.exists():
                scenario_stats[scenario_name] = pd.read_csv(stats_path)
        
        # Baseline forecasts for comparison
        forecast_path = self.output_path / 'forecast_comparison.csv'
        forecasts_df = pd.read_csv(forecast_path) if forecast_path.exists() else None
        
        # Uncertainty decomposition
        decomp_path = self.output_path / 'uncertainty_decomposition.csv'
        decomp_df = pd.read_csv(decomp_path) if decomp_path.exists() else None
        
        return timeline_df, scenario_stats, forecasts_df, decomp_df
    
    def plot_scenario_trajectories(self, scenario_stats):
        """Plot projected trajectories for all scenarios"""
        print("Plotting scenario trajectories...")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Generate dates for x-axis
        start_date = pd.to_datetime('2025-11-01')
        
        colors = ['blue', 'orange', 'red', 'green']
        
        for idx, (scenario_name, stats_df) in enumerate(scenario_stats.items()):
            # Create date column
            dates = start_date + pd.to_timedelta(stats_df['month'] * 30.44, unit='D')
            
            scenario_config = self.scenarios[scenario_name]
            label = scenario_config['name']
            color = colors[idx % len(colors)]
            
            # Plot median trajectory
            ax.plot(dates, stats_df['median'], label=label, linewidth=2.5, color=color)
            
            # Plot confidence interval
            ax.fill_between(dates, stats_df['p10'], stats_df['p90'],
                           alpha=0.2, color=color)
        
        # AGI threshold line
        ax.axhline(90, color='red', linestyle='--', linewidth=2, 
                  label='AGI Threshold (90%)', alpha=0.7)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Benchmark Score', fontsize=12)
        ax.set_title('AI Capability Trajectories Under Policy Scenarios',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(50, 100)
        
        plt.tight_layout()
        
        save_path = self.output_path / 'scenario_trajectories.png'
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
    
    def plot_timeline_comparison(self, timeline_df, forecasts_df):
        """Plot AGI timeline estimates comparison"""
        print("Plotting timeline comparison...")
        
        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0], self.figsize[1]*0.8))
        
        # Plot 1: ExogenousAI scenarios
        scenarios = timeline_df['scenario'].tolist()
        medians = timeline_df['median_year'].tolist()
        p10s = timeline_df['p10_year'].tolist()
        p90s = timeline_df['p90_year'].tolist()
        
        y_pos = np.arange(len(scenarios))
        
        axes[0].barh(y_pos, medians, color='skyblue', alpha=0.7, edgecolor='black')
        
        # Error bars for confidence intervals
        xerr = [[m - p10 for m, p10 in zip(medians, p10s)],
                [p90 - m for m, p90 in zip(medians, p90s)]]
        
        axes[0].errorbar(medians, y_pos, xerr=xerr, fmt='none', color='black',
                        capsize=5, capthick=2)
        
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(scenarios, fontsize=10)
        axes[0].set_xlabel('Year', fontsize=12)
        axes[0].set_title('ExogenousAI Policy Scenarios', fontsize=12, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Plot 2: All forecasts comparison
        if forecasts_df is not None:
            all_sources = forecasts_df['source'].tolist()
            all_medians = forecasts_df['median_year'].tolist()
            
            # Color by method
            colors = ['green' if 'ExogenousAI' in s else 'gray' for s in all_sources]
            
            y_pos2 = np.arange(len(all_sources))
            
            axes[1].barh(y_pos2, all_medians, color=colors, alpha=0.7, edgecolor='black')
            axes[1].set_yticks(y_pos2)
            axes[1].set_yticklabels(all_sources, fontsize=9)
            axes[1].set_xlabel('Year', fontsize=12)
            axes[1].set_title('Comparison with Literature Baselines', 
                            fontsize=12, fontweight='bold')
            axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_path / 'timeline_comparison.png'
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
    
    def plot_uncertainty_decomposition(self, decomp_df):
        """Plot pie chart of uncertainty decomposition"""
        if decomp_df is None:
            return
        
        print("Plotting uncertainty decomposition...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract proportions
        technical = decomp_df['technical_proportion'].iloc[0] * 100
        economic = decomp_df['economic_proportion'].iloc[0] * 100
        policy = decomp_df['policy_proportion'].iloc[0] * 100
        
        sizes = [technical, economic, policy]
        labels = [f'Technical\nUncertainty\n({technical:.1f}%)',
                 f'Economic\nUncertainty\n({economic:.1f}%)',
                 f'Policy\nUncertainty\n({policy:.1f}%)']
        
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        explode = (0.05, 0.05, 0.1)  # Emphasize policy
        
        ax.pie(sizes, explode=explode, labels=labels, colors=colors,
              autopct='', shadow=True, startangle=90,
              textprops={'fontsize': 12, 'fontweight': 'bold'})
        
        ax.set_title('Decomposition of AGI Timeline Uncertainty',
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        save_path = self.output_path / 'uncertainty_decomposition.png'
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
    
    def plot_distribution_violin(self, scenario_stats, timeline_df):
        """Plot violin plot of AGI timeline distributions"""
        print("Plotting timeline distributions...")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Prepare data for violin plot
        data_for_violin = []
        labels_for_violin = []
        
        for _, row in timeline_df.iterrows():
            # Approximate distribution from percentiles
            # Generate synthetic data matching the percentiles
            n_samples = 1000
            
            # Use beta distribution to approximate
            median = row['median_year']
            p10 = row['p10_year']
            p90 = row['p90_year']
            
            # Generate normal distribution matching percentiles
            mean = median
            std = (p90 - p10) / 2.56  # Approximation for 80% CI
            
            samples = np.random.normal(mean, std, n_samples)
            samples = np.clip(samples, p10 - 5, p90 + 5)
            
            data_for_violin.append(samples)
            labels_for_violin.append(row['scenario'])
        
        # Create violin plot
        parts = ax.violinplot(data_for_violin, positions=range(len(labels_for_violin)),
                             showmedians=True, widths=0.7)
        
        # Customize colors
        for pc in parts['bodies']:
            pc.set_facecolor('skyblue')
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(labels_for_violin)))
        ax.set_xticklabels(labels_for_violin, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Estimated AGI Year', fontsize=12)
        ax.set_title('Distribution of AGI Timeline Estimates by Scenario',
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_path / 'timeline_distributions.png'
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
    
    def run(self):
        """Execute full plotting pipeline"""
        print("="*60)
        print("Starting Scenario Visualization")
        print("="*60)
        
        # Load data
        timeline_df, scenario_stats, forecasts_df, decomp_df = self.load_data()
        
        # Create plots
        self.plot_scenario_trajectories(scenario_stats)
        self.plot_timeline_comparison(timeline_df, forecasts_df)
        self.plot_uncertainty_decomposition(decomp_df)
        self.plot_distribution_violin(scenario_stats, timeline_df)
        
        print(f"\n✓ Scenario visualization complete!")


if __name__ == "__main__":
    plotter = ScenarioPlotter()
    plotter.run()
