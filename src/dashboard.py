"""
Interactive Dashboard for ExogenousAI
Streamlit application for exploring policy impact on AI timelines
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yaml
from pathlib import Path


# Page configuration
st.set_page_config(
    page_title="ExogenousAI Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


class ExogenousAIDashboard:
    """Interactive dashboard for ExogenousAI analysis"""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize dashboard"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_path = Path(self.config['paths']['output'])
        self.processed_path = Path(self.config['paths']['processed_data'])
        
        self.load_data()
    
    def load_data(self):
        """Load all analysis results"""
        # Timeline estimates
        self.timeline_df = pd.read_csv(self.output_path / 'agi_timeline_estimates.csv')
        
        # Event study results
        self.event_results_df = pd.read_csv(self.output_path / 'event_study_results.csv')
        
        # Forecast comparison
        self.forecasts_df = pd.read_csv(self.output_path / 'forecast_comparison.csv')
        
        # Uncertainty decomposition
        self.decomp_df = pd.read_csv(self.output_path / 'uncertainty_decomposition.csv')
        
        # Scenario statistics
        self.scenario_stats = {}
        for scenario_name in self.config['scenarios'].keys():
            path = self.output_path / f'scenario_{scenario_name}_statistics.csv'
            if path.exists():
                self.scenario_stats[scenario_name] = pd.read_csv(path)
        
        # Historical data
        self.historical_df = pd.read_csv(self.processed_path / 'merged_dataset.csv')
        self.historical_df['date'] = pd.to_datetime(self.historical_df['date'])
    
    def render_header(self):
        """Render dashboard header"""
        st.title("🤖 ExogenousAI: Policy Impact Dashboard")
        st.markdown("### Policy Shock Impact Analysis on AI Capability Timelines")
        
        st.markdown("""
        This dashboard presents analysis of how policy interventions (export controls, compute governance)
        quantitatively shift probabilistic AI capability forecasts.
        """)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            median_range = f"{int(self.timeline_df['median_year'].min())}-{int(self.timeline_df['median_year'].max())}"
            st.metric("Timeline Range", median_range, help="Range of median AGI estimates")
        
        with col2:
            sig_count = self.event_results_df['is_significant'].sum()
            total = len(self.event_results_df)
            st.metric("Significant Events", f"{sig_count}/{total}", 
                     help="Events with statistically significant impact")
        
        with col3:
            policy_contrib = self.decomp_df['policy_proportion'].iloc[0] * 100
            st.metric("Policy Uncertainty", f"{policy_contrib:.1f}%",
                     help="Proportion of uncertainty from policy")
        
        with col4:
            timeline_spread = int(self.timeline_df['median_year'].max() - 
                                self.timeline_df['median_year'].min())
            st.metric("Scenario Spread", f"{timeline_spread} years",
                     help="Difference between fastest/slowest scenarios")
    
    def render_scenario_explorer(self):
        """Render interactive scenario explorer"""
        st.header("📊 Scenario Explorer")
        
        # Sidebar controls
        with st.sidebar:
            st.subheader("Scenario Parameters")
            
            selected_scenarios = st.multiselect(
                "Select Scenarios",
                options=self.timeline_df['scenario'].tolist(),
                default=self.timeline_df['scenario'].tolist()
            )
            
            show_confidence = st.checkbox("Show Confidence Intervals", value=True)
            agi_threshold = st.slider("AGI Threshold", 85.0, 95.0, 90.0, 0.5)
        
        # Filter data
        filtered_timeline = self.timeline_df[self.timeline_df['scenario'].isin(selected_scenarios)]
        
        # Plot trajectory projections
        fig = go.Figure()
        
        start_date = pd.to_datetime('2025-11-01')
        
        for scenario in selected_scenarios:
            scenario_key = [k for k, v in self.config['scenarios'].items() 
                           if v['name'] == scenario][0]
            
            if scenario_key in self.scenario_stats:
                stats_df = self.scenario_stats[scenario_key]
                dates = start_date + pd.to_timedelta(stats_df['month'] * 30.44, unit='D')
                
                # Median line
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=stats_df['median'],
                    mode='lines',
                    name=scenario,
                    line=dict(width=3)
                ))
                
                # Confidence interval
                if show_confidence:
                    fig.add_trace(go.Scatter(
                        x=dates.tolist() + dates.tolist()[::-1],
                        y=stats_df['p90'].tolist() + stats_df['p10'].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(0,100,200,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name=f'{scenario} CI',
                        showlegend=False
                    ))
        
        # AGI threshold line
        fig.add_hline(y=agi_threshold, line_dash="dash", line_color="red",
                     annotation_text=f"AGI Threshold ({agi_threshold}%)")
        
        fig.update_layout(
            title="AI Capability Trajectories Under Policy Scenarios",
            xaxis_title="Date",
            yaxis_title="Benchmark Score",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Timeline comparison table
        st.subheader("Timeline Estimates")
        
        display_df = filtered_timeline[['scenario', 'median_year', 'p10_year', 'p90_year']].copy()
        display_df.columns = ['Scenario', 'Median Year', '10th Percentile', '90th Percentile']
        display_df = display_df.round(0).astype(int, errors='ignore')
        
        st.dataframe(display_df, use_container_width=True)
    
    def render_event_study(self):
        """Render event study results"""
        st.header("📈 Event Study Analysis")
        
        # Event selector
        selected_metric = st.selectbox(
            "Select Metric",
            options=self.event_results_df['metric'].unique()
        )
        
        # Filter by metric
        metric_results = self.event_results_df[
            self.event_results_df['metric'] == selected_metric
        ].sort_values('abnormal_return_change')
        
        # Bar chart of abnormal returns
        fig = px.bar(
            metric_results,
            x='abnormal_return_change',
            y='event_name',
            orientation='h',
            color='is_significant',
            color_discrete_map={True: 'red', False: 'gray'},
            labels={
                'abnormal_return_change': 'Abnormal Return (%)',
                'event_name': 'Policy Event',
                'is_significant': 'Significant'
            },
            title=f"Policy Event Impact on {selected_metric}"
        )
        
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        fig.update_layout(height=400, showlegend=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results table
        st.subheader("Detailed Event Analysis")
        
        display_cols = ['event_name', 'abnormal_return_change', 't_statistic', 
                       'p_value', 'is_significant']
        
        display_df = metric_results[display_cols].copy()
        display_df.columns = ['Event', 'Abnormal Return (%)', 't-statistic', 
                             'p-value', 'Significant']
        display_df['Abnormal Return (%)'] = display_df['Abnormal Return (%)'].round(2)
        display_df['t-statistic'] = display_df['t-statistic'].round(3)
        display_df['p-value'] = display_df['p-value'].round(4)
        
        st.dataframe(display_df, use_container_width=True)
    
    def render_forecast_comparison(self):
        """Render comparison with other forecasts"""
        st.header("🔮 Forecast Comparison")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Timeline comparison chart
            fig = px.bar(
                self.forecasts_df.sort_values('median_year'),
                x='median_year',
                y='source',
                orientation='h',
                color='method',
                labels={
                    'median_year': 'Median AGI Year',
                    'source': 'Forecast Source',
                    'method': 'Method'
                },
                title="AGI Timeline Estimates: Literature Comparison"
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Uncertainty decomposition pie chart
            st.subheader("Uncertainty Sources")
            
            decomp_data = {
                'Source': ['Technical', 'Economic', 'Policy'],
                'Proportion': [
                    self.decomp_df['technical_proportion'].iloc[0] * 100,
                    self.decomp_df['economic_proportion'].iloc[0] * 100,
                    self.decomp_df['policy_proportion'].iloc[0] * 100
                ]
            }
            
            fig = px.pie(
                decomp_data,
                values='Proportion',
                names='Source',
                title='Uncertainty Decomposition',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Full comparison table
        st.subheader("All Forecasts")
        
        display_df = self.forecasts_df[['source', 'model', 'median_year', 
                                        'p10_year', 'p90_year']].copy()
        display_df.columns = ['Source', 'Model', 'Median Year', 
                             'P10 Year', 'P90 Year']
        
        st.dataframe(display_df, use_container_width=True)
    
    def render_historical_data(self):
        """Render historical data exploration"""
        st.header("📚 Historical Data")
        
        # Metric selector
        available_metrics = [col for col in self.historical_df.columns 
                           if col.startswith(('benchmark_', 'arxiv_', 'nvda_'))]
        
        selected_metric = st.selectbox(
            "Select Metric to Visualize",
            options=available_metrics
        )
        
        # Time series plot
        fig = go.Figure()
        
        # Main metric line
        fig.add_trace(go.Scatter(
            x=self.historical_df['date'],
            y=self.historical_df[selected_metric],
            mode='lines+markers',
            name=selected_metric,
            line=dict(width=2)
        ))
        
        # Add event markers
        events = self.historical_df[self.historical_df['event_period'] == 'event']
        events = events.drop_duplicates(subset=['event_name'])
        
        for _, event in events.iterrows():
            fig.add_vline(
                x=event['date'],
                line_dash="dash",
                line_color="red",
                annotation_text=event['event_name'],
                annotation_position="top"
            )
        
        fig.update_layout(
            title=f"Historical {selected_metric} with Policy Events",
            xaxis_title="Date",
            yaxis_title=selected_metric,
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean", f"{self.historical_df[selected_metric].mean():.2f}")
        
        with col2:
            st.metric("Std Dev", f"{self.historical_df[selected_metric].std():.2f}")
        
        with col3:
            st.metric("Min", f"{self.historical_df[selected_metric].min():.2f}")
        
        with col4:
            st.metric("Max", f"{self.historical_df[selected_metric].max():.2f}")
    
    def render_about(self):
        """Render about section"""
        with st.sidebar:
            st.markdown("---")
            st.subheader("About")
            st.markdown("""
            **ExogenousAI** analyzes how policy interventions impact AI capability timelines
            using event study methodology and Monte Carlo simulation.
            
            **Methodology:**
            - Event study analysis
            - Monte Carlo simulation
            - Uncertainty decomposition
            
            **Data Sources:**
            - Papers With Code benchmarks
            - arXiv submission counts
            - NVIDIA stock data
            - Policy event timeline
            """)
    
    def run(self):
        """Run the dashboard"""
        self.render_header()
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Scenarios",
            "📈 Event Study",
            "🔮 Forecast Comparison",
            "📚 Historical Data"
        ])
        
        with tab1:
            self.render_scenario_explorer()
        
        with tab2:
            self.render_event_study()
        
        with tab3:
            self.render_forecast_comparison()
        
        with tab4:
            self.render_historical_data()
        
        self.render_about()


if __name__ == "__main__":
    dashboard = ExogenousAIDashboard()
    dashboard.run()
