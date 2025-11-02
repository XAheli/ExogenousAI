"""
Event Study Visualization
Creates plots for event study results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path


class EventPlotter:
    """Create visualizations for event study analysis"""
    
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
    
    def load_data(self):
        """Load event study results and data"""
        print("Loading data...")
        
        # Event study results
        results_path = self.output_path / 'event_study_results.csv'
        results_df = pd.read_csv(results_path)
        
        # CAR data
        car_path = self.output_path / 'cumulative_abnormal_returns.csv'
        car_df = pd.read_csv(car_path)
        car_df['date'] = pd.to_datetime(car_df['date'])
        
        # Merged dataset
        data_path = Path(self.config['paths']['processed_data']) / 'merged_dataset.csv'
        data_df = pd.read_csv(data_path)
        data_df['date'] = pd.to_datetime(data_df['date'])
        
        return results_df, car_df, data_df
    
    def plot_abnormal_returns_by_event(self, results_df):
        """Plot abnormal returns for each event"""
        print("Plotting abnormal returns by event...")
        
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, height_ratios=[2, 1])
        
        # Group by event
        events = results_df.groupby(['event_name', 'event_id']).agg({
            'abnormal_return_change': 'mean',
            'is_significant': 'any'
        }).reset_index()
        
        events = events.sort_values('abnormal_return_change')
        
        # Color by significance
        colors = ['red' if sig else 'gray' for sig in events['is_significant']]
        
        # Bar plot
        axes[0].barh(range(len(events)), events['abnormal_return_change'], color=colors)
        axes[0].set_yticks(range(len(events)))
        axes[0].set_yticklabels(events['event_name'], fontsize=10)
        axes[0].set_xlabel('Average Abnormal Return (%)', fontsize=12)
        axes[0].set_title('Policy Event Impact on AI Capabilities', fontsize=14, fontweight='bold')
        axes[0].axvline(0, color='black', linestyle='--', alpha=0.5)
        axes[0].grid(axis='x', alpha=0.3)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Significant (p < 0.05)'),
            Patch(facecolor='gray', label='Not Significant')
        ]
        axes[0].legend(handles=legend_elements, loc='lower right')
        
        # By metric heatmap
        pivot = results_df.pivot_table(
            index='event_name',
            columns='metric',
            values='abnormal_return_change',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Abnormal Return (%)'}, ax=axes[1])
        axes[1].set_title('Abnormal Returns by Event and Metric', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('')
        axes[1].set_xlabel('')
        
        plt.tight_layout()
        
        save_path = self.output_path / 'abnormal_returns_by_event.png'
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
    
    def plot_cumulative_abnormal_returns(self, car_df):
        """Plot cumulative abnormal returns over event windows"""
        print("Plotting cumulative abnormal returns...")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot CAR for each event
        for event_id in car_df['event_id'].unique():
            event_data = car_df[car_df['event_id'] == event_id]
            event_name = event_data['event_name'].iloc[0]
            
            ax.plot(event_data['months_from_event'], event_data['CAR'],
                   marker='o', label=event_name, linewidth=2)
        
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='Event Month')
        
        ax.set_xlabel('Months from Event', fontsize=12)
        ax.set_ylabel('Cumulative Abnormal Return (%)', fontsize=12)
        ax.set_title('Cumulative Abnormal Returns Around Policy Events', 
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_path / 'cumulative_abnormal_returns.png'
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
    
    def plot_time_series_with_events(self, data_df):
        """Plot time series of AI metrics with event markers"""
        print("Plotting time series with event markers...")
        
        fig, axes = plt.subplots(3, 1, figsize=(self.figsize[0], self.figsize[1]*1.2))
        
        # Get event dates
        events = data_df[data_df['event_period'] == 'event'].dropna(subset=['event_name'])
        events = events.drop_duplicates(subset=['event_name'])
        
        # Plot 1: Benchmark Composite
        if 'benchmark_composite' in data_df.columns:
            axes[0].plot(data_df['date'], data_df['benchmark_composite'], 
                        linewidth=2, label='Benchmark Composite')
            
            for _, event in events.iterrows():
                axes[0].axvline(event['date'], color='red', linestyle='--', alpha=0.6)
            
            axes[0].set_ylabel('Benchmark Score', fontsize=11)
            axes[0].set_title('AI Benchmark Progress with Policy Events', 
                            fontsize=14, fontweight='bold')
            axes[0].grid(alpha=0.3)
            axes[0].legend()
        
        # Plot 2: Research Velocity
        if 'research_velocity' in data_df.columns:
            axes[1].plot(data_df['date'], data_df['research_velocity'], 
                        linewidth=2, label='arXiv Submissions', color='green')
            
            for _, event in events.iterrows():
                axes[1].axvline(event['date'], color='red', linestyle='--', alpha=0.6)
            
            axes[1].set_ylabel('Monthly Submissions', fontsize=11)
            axes[1].set_title('Research Velocity', fontsize=12)
            axes[1].grid(alpha=0.3)
            axes[1].legend()
        
        # Plot 3: Investment Sentiment (NVDA)
        if 'nvda_close' in data_df.columns:
            axes[2].plot(data_df['date'], data_df['nvda_close'], 
                        linewidth=2, label='NVIDIA Stock Price', color='purple')
            
            for _, event in events.iterrows():
                axes[2].axvline(event['date'], color='red', linestyle='--', alpha=0.6,
                              label='Policy Event' if _ == events.index[0] else '')
            
            axes[2].set_ylabel('Stock Price ($)', fontsize=11)
            axes[2].set_xlabel('Date', fontsize=11)
            axes[2].set_title('Investment Sentiment Proxy', fontsize=12)
            axes[2].grid(alpha=0.3)
            axes[2].legend()
        
        plt.tight_layout()
        
        save_path = self.output_path / 'time_series_with_events.png'
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
    
    def plot_statistical_significance(self, results_df):
        """Plot statistical significance of results"""
        print("Plotting statistical significance...")
        
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Scatter: AR vs p-value
        axes[0].scatter(results_df['abnormal_return_change'], results_df['p_value'],
                       c=results_df['is_significant'], cmap='RdYlGn_r',
                       s=100, alpha=0.7, edgecolors='black')
        
        axes[0].axhline(0.05, color='red', linestyle='--', label='p = 0.05')
        axes[0].axvline(0, color='gray', linestyle='--', alpha=0.5)
        
        axes[0].set_xlabel('Abnormal Return (%)', fontsize=12)
        axes[0].set_ylabel('P-value', fontsize=12)
        axes[0].set_title('Statistical Significance of Event Impacts', fontsize=12, fontweight='bold')
        axes[0].set_yscale('log')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Bar: Significance count by metric
        sig_count = results_df.groupby('metric')['is_significant'].agg(['sum', 'count'])
        sig_count['not_sig'] = sig_count['count'] - sig_count['sum']
        
        sig_count[['sum', 'not_sig']].plot(kind='bar', stacked=True, ax=axes[1],
                                          color=['green', 'red'], alpha=0.7)
        
        axes[1].set_xlabel('Metric', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('Significance by Metric', fontsize=12, fontweight='bold')
        axes[1].set_xticklabels(sig_count.index, rotation=45, ha='right')
        axes[1].legend(['Significant', 'Not Significant'])
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_path / 'statistical_significance.png'
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        print(f"  Saved to {save_path}")
        plt.close()
    
    def run(self):
        """Execute full plotting pipeline"""
        print("="*60)
        print("Starting Event Study Visualization")
        print("="*60)
        
        # Load data
        results_df, car_df, data_df = self.load_data()
        
        # Create plots
        self.plot_abnormal_returns_by_event(results_df)
        self.plot_cumulative_abnormal_returns(car_df)
        self.plot_time_series_with_events(data_df)
        self.plot_statistical_significance(results_df)
        
        print(f"\n✓ Event visualization complete!")


if __name__ == "__main__":
    plotter = EventPlotter()
    plotter.run()
