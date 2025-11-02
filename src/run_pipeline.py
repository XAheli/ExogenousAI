#!/usr/bin/env python3
"""
Main Pipeline Runner
Executes the complete ExogenousAI analysis pipeline
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_collection.scrape_pwc import PWCScraper
from data_collection.scrape_arxiv import ArxivScraper
from data_collection.scrape_stocks import StockScraper
from data_collection.compile_policy_events import PolicyEventCompiler
from preprocessing.clean_benchmarks import BenchmarkCleaner
from preprocessing.aggregate_monthly import MonthlyAggregator
from preprocessing.merge_datasets import DatasetMerger
from analysis.event_study import EventStudyAnalyzer
from analysis.monte_carlo import MonteCarloSimulator
from analysis.meta_forecasting import MetaForecastingAnalyzer
from visualization.plot_events import EventPlotter
from visualization.plot_scenarios import ScenarioPlotter


def run_pipeline():
    """Execute full analysis pipeline"""
    
    print("\n" + "="*70)
    print(" "*15 + "EXOGENOUSAI ANALYSIS PIPELINE")
    print("="*70 + "\n")
    
    try:
        # Phase 1: Data Collection
        print("\n🔍 PHASE 1: DATA COLLECTION")
        print("-" * 70)
        
        print("\n[1/4] Collecting benchmark data...")
        pwc_scraper = PWCScraper()
        pwc_scraper.run()
        
        print("\n[2/4] Collecting arXiv data...")
        arxiv_scraper = ArxivScraper()
        arxiv_scraper.run()
        
        print("\n[3/4] Collecting stock data...")
        stock_scraper = StockScraper()
        stock_scraper.run()
        
        print("\n[4/4] Compiling policy events...")
        policy_compiler = PolicyEventCompiler()
        policy_compiler.run()
        
        # Phase 2: Data Preprocessing
        print("\n\n🔧 PHASE 2: DATA PREPROCESSING")
        print("-" * 70)
        
        print("\n[1/3] Cleaning benchmark data...")
        cleaner = BenchmarkCleaner()
        cleaner.run()
        
        print("\n[2/3] Aggregating to monthly...")
        aggregator = MonthlyAggregator()
        aggregator.run()
        
        print("\n[3/3] Merging all datasets...")
        merger = DatasetMerger()
        merger.run()
        
        # Phase 3: Analysis
        print("\n\n📊 PHASE 3: ANALYSIS")
        print("-" * 70)
        
        print("\n[1/3] Running event study analysis...")
        event_analyzer = EventStudyAnalyzer()
        event_analyzer.run()
        
        print("\n[2/3] Running Monte Carlo simulation...")
        mc_simulator = MonteCarloSimulator()
        mc_simulator.run()
        
        print("\n[3/3] Running meta-forecasting analysis...")
        meta_analyzer = MetaForecastingAnalyzer()
        meta_analyzer.run()
        
        # Phase 4: Visualization
        print("\n\n📈 PHASE 4: VISUALIZATION")
        print("-" * 70)
        
        print("\n[1/2] Creating event study plots...")
        event_plotter = EventPlotter()
        event_plotter.run()
        
        print("\n[2/2] Creating scenario plots...")
        scenario_plotter = ScenarioPlotter()
        scenario_plotter.run()
        
        # Success
        print("\n\n" + "="*70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        print("\n📁 Output files located in:")
        print(f"   - data/processed/")
        print(f"   - data/output/")
        
        print("\n🚀 To launch the dashboard, run:")
        print(f"   streamlit run src/dashboard.py")
        
        return True
        
    except Exception as e:
        print(f"\n\n❌ PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
