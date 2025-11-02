"""
Test Suite for ExogenousAI
Unit tests for core functionality
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestDataCollection(unittest.TestCase):
    """Test data collection modules"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temp files"""
        shutil.rmtree(self.temp_dir)
    
    def test_pwc_scraper_init(self):
        """Test PWC scraper initialization"""
        from data_collection.scrape_pwc import PWCScraper
        scraper = PWCScraper()
        
        self.assertIsNotNone(scraper.benchmarks)
        self.assertGreater(scraper.top_k, 0)
    
    def test_arxiv_scraper_init(self):
        """Test arXiv scraper initialization"""
        from data_collection.scrape_arxiv import ArxivScraper
        scraper = ArxivScraper()
        
        self.assertIsNotNone(scraper.categories)
        self.assertGreater(len(scraper.categories), 0)
    
    def test_stock_scraper_init(self):
        """Test stock scraper initialization"""
        from data_collection.scrape_stocks import StockScraper
        scraper = StockScraper()
        
        self.assertEqual(scraper.ticker, 'NVDA')


class TestPreprocessing(unittest.TestCase):
    """Test preprocessing modules"""
    
    def test_benchmark_cleaner_init(self):
        """Test benchmark cleaner initialization"""
        from preprocessing.clean_benchmarks import BenchmarkCleaner
        cleaner = BenchmarkCleaner()
        
        self.assertIsNotNone(cleaner.raw_path)
        self.assertIsNotNone(cleaner.processed_path)
    
    def test_data_validation(self):
        """Test data validation"""
        # Create test data
        test_data = pd.DataFrame({
            'date': pd.date_range('2022-01-01', periods=10, freq='MS'),
            'benchmark': ['MMLU'] * 10,
            'score': np.random.uniform(50, 90, 10),
            'rank': range(1, 11)
        })
        
        # Test score range
        self.assertTrue((test_data['score'] >= 0).all())
        self.assertTrue((test_data['score'] <= 100).all())


class TestAnalysis(unittest.TestCase):
    """Test analysis modules"""
    
    def test_event_study_init(self):
        """Test event study analyzer initialization"""
        from analysis.event_study import EventStudyAnalyzer
        analyzer = EventStudyAnalyzer()
        
        self.assertGreater(analyzer.pre_window, 0)
        self.assertGreater(analyzer.post_window, 0)
    
    def test_monte_carlo_init(self):
        """Test Monte Carlo simulator initialization"""
        from analysis.monte_carlo import MonteCarloSimulator
        simulator = MonteCarloSimulator()
        
        self.assertGreater(simulator.n_iterations, 0)
        self.assertGreater(simulator.forecast_horizon, 0)
    
    def test_exponential_fit(self):
        """Test exponential trend fitting"""
        # Generate synthetic exponential data
        x = np.arange(20)
        y = 50 * np.exp(0.02 * x) + np.random.normal(0, 1, 20)
        
        # Fit linear to log-transformed data
        log_y = np.log(y)
        coeffs = np.polyfit(x, log_y, 1)
        
        # Check that growth rate is positive
        self.assertGreater(coeffs[0], 0)


class TestVisualization(unittest.TestCase):
    """Test visualization modules"""
    
    def test_event_plotter_init(self):
        """Test event plotter initialization"""
        from visualization.plot_events import EventPlotter
        plotter = EventPlotter()
        
        self.assertIsNotNone(plotter.figsize)
        self.assertGreater(plotter.dpi, 0)
    
    def test_scenario_plotter_init(self):
        """Test scenario plotter initialization"""
        from visualization.plot_scenarios import ScenarioPlotter
        plotter = ScenarioPlotter()
        
        self.assertIsNotNone(plotter.scenarios)


class TestConfiguration(unittest.TestCase):
    """Test configuration loading"""
    
    def test_config_load(self):
        """Test configuration file loading"""
        import yaml
        
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        self.assertIn('project', config)
        self.assertIn('paths', config)
        self.assertIn('data_collection', config)
        self.assertIn('scenarios', config)
    
    def test_policy_events(self):
        """Test policy events configuration"""
        import yaml
        
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        events = config['policy_events']['events']
        
        self.assertGreater(len(events), 0)
        
        for event in events:
            self.assertIn('date', event)
            self.assertIn('name', event)
            self.assertIn('severity', event)


class TestUtilities(unittest.TestCase):
    """Test utility functions"""
    
    def test_date_parsing(self):
        """Test date parsing"""
        date_str = '2022-10-07'
        date = pd.to_datetime(date_str)
        
        self.assertEqual(date.year, 2022)
        self.assertEqual(date.month, 10)
        self.assertEqual(date.day, 7)
    
    def test_percentile_calculation(self):
        """Test percentile calculations"""
        data = np.random.normal(100, 15, 1000)
        
        p10 = np.percentile(data, 10)
        p50 = np.percentile(data, 50)
        p90 = np.percentile(data, 90)
        
        self.assertLess(p10, p50)
        self.assertLess(p50, p90)


if __name__ == '__main__':
    unittest.main()
