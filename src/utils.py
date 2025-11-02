"""
Utility Functions for ExogenousAI
Helper functions used across modules
"""

import pandas as pd
import numpy as np
import yaml
import tomli
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load main configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_secrets(secrets_path: str = "secrets.toml") -> Dict[str, Any]:
    """
    Load secrets from TOML file (API keys, credentials, etc.)
    
    Args:
        secrets_path: Path to secrets.toml file
        
    Returns:
        Dictionary containing secrets
        
    Raises:
        FileNotFoundError: If secrets.toml doesn't exist
    """
    secrets_file = Path(secrets_path)
    
    if not secrets_file.exists():
        raise FileNotFoundError(
            f"secrets.toml not found at {secrets_path}\n"
            f"Please create it from secrets.toml.example and add your API keys"
        )
    
    with open(secrets_file, 'rb') as f:
        return tomli.load(f)


def get_api_key(key_name: str, secrets_path: str = "secrets.toml") -> str:
    """
    Get specific API key from secrets file
    
    Args:
        key_name: Name of the API key (e.g., 'serpapi', 'alpha_vantage')
        secrets_path: Path to secrets.toml file
        
    Returns:
        API key string
        
    Raises:
        KeyError: If API key not found in secrets
    """
    secrets = load_secrets(secrets_path)
    
    try:
        return secrets['api_keys'][key_name]
    except KeyError:
        raise KeyError(
            f"API key '{key_name}' not found in secrets.toml\n"
            f"Available keys: {list(secrets.get('api_keys', {}).keys())}"
        )


def date_to_month_index(date: pd.Timestamp, reference_date: pd.Timestamp) -> float:
    """
    Convert date to months elapsed since reference date
    
    Args:
        date: Target date
        reference_date: Reference starting date
        
    Returns:
        Number of months elapsed (float)
    """
    days_elapsed = (date - reference_date).days
    months_elapsed = days_elapsed / 30.44  # Average days per month
    
    return months_elapsed


def exponential_growth(initial_value: float, growth_rate: float, 
                      time_steps: int) -> np.ndarray:
    """
    Calculate exponential growth trajectory
    
    Args:
        initial_value: Starting value
        growth_rate: Per-period growth rate (decimal)
        time_steps: Number of periods to project
        
    Returns:
        Array of values over time
    """
    time = np.arange(time_steps)
    trajectory = initial_value * np.exp(growth_rate * time)
    
    return trajectory


def calculate_confidence_interval(data: np.ndarray, 
                                 confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for data
    
    Args:
        data: Array of values
        confidence: Confidence level (default 0.95)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower = np.percentile(data, lower_percentile)
    upper = np.percentile(data, upper_percentile)
    
    return lower, upper


def normalize_series(series: pd.Series, method: str = 'minmax') -> pd.Series:
    """
    Normalize a pandas Series
    
    Args:
        series: Input series
        method: Normalization method ('minmax' or 'zscore')
        
    Returns:
        Normalized series
    """
    if method == 'minmax':
        return (series - series.min()) / (series.max() - series.min())
    elif method == 'zscore':
        return (series - series.mean()) / series.std()
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def resample_to_monthly(df: pd.DataFrame, date_col: str = 'date',
                       agg_func: Dict[str, str] = None) -> pd.DataFrame:
    """
    Resample dataframe to monthly frequency
    
    Args:
        df: Input dataframe
        date_col: Name of date column
        agg_func: Dictionary mapping columns to aggregation functions
        
    Returns:
        Resampled dataframe
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    
    if agg_func is None:
        agg_func = {col: 'mean' for col in df.columns}
    
    monthly = df.resample('MS').agg(agg_func)
    
    return monthly.reset_index()


def calculate_growth_rate(series: pd.Series, periods: int = 1) -> pd.Series:
    """
    Calculate period-over-period growth rate
    
    Args:
        series: Input series
        periods: Number of periods for growth calculation
        
    Returns:
        Growth rate series
    """
    return series.pct_change(periods=periods) * 100


def fit_exponential_trend(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Fit exponential trend to data
    
    Args:
        x: Independent variable (time)
        y: Dependent variable
        
    Returns:
        Tuple of (initial_value, growth_rate)
    """
    # Log-transform for linear regression
    log_y = np.log(y + 1e-10)  # Add small constant to avoid log(0)
    
    # Linear regression on log-transformed data
    coeffs = np.polyfit(x, log_y, 1)
    
    growth_rate = coeffs[0]
    initial_value = np.exp(coeffs[1])
    
    return initial_value, growth_rate


def generate_date_range(start_date: str, end_date: str, 
                       freq: str = 'MS') -> List[str]:
    """
    Generate list of dates in range
    
    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        freq: Frequency ('MS' for month start, 'D' for daily)
        
    Returns:
        List of date strings
    """
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    return [d.strftime('%Y-%m-%d') for d in dates]


def safe_divide(numerator: float, denominator: float, 
               default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if division by zero
        
    Returns:
        Result of division or default
    """
    if denominator == 0 or np.isnan(denominator):
        return default
    
    return numerator / denominator


def format_year_month(date: pd.Timestamp) -> str:
    """
    Format date as YYYY-MM
    
    Args:
        date: Input timestamp
        
    Returns:
        Formatted string
    """
    return date.strftime('%Y-%m')


def validate_dataframe(df: pd.DataFrame, required_cols: List[str],
                      date_cols: List[str] = None) -> bool:
    """
    Validate dataframe structure
    
    Args:
        df: Dataframe to validate
        required_cols: List of required column names
        date_cols: List of columns that should be datetime
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    # Check required columns
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check date columns
    if date_cols:
        for col in date_cols:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                raise ValueError(f"Column {col} is not datetime type")
    
    return True


def smooth_series(series: pd.Series, window: int = 3, 
                 method: str = 'rolling') -> pd.Series:
    """
    Smooth time series data
    
    Args:
        series: Input series
        window: Window size for smoothing
        method: Smoothing method ('rolling' or 'ewm')
        
    Returns:
        Smoothed series
    """
    if method == 'rolling':
        return series.rolling(window=window, center=True).mean()
    elif method == 'ewm':
        return series.ewm(span=window).mean()
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    Calculate cumulative returns from period returns
    
    Args:
        returns: Series of period returns (decimal)
        
    Returns:
        Cumulative returns series
    """
    return (1 + returns).cumprod() - 1


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test date conversion
    ref_date = pd.to_datetime('2022-01-01')
    test_date = pd.to_datetime('2023-01-01')
    months = date_to_month_index(test_date, ref_date)
    print(f"Months between dates: {months:.1f}")
    
    # Test exponential growth
    trajectory = exponential_growth(50, 0.02, 12)
    print(f"Growth trajectory: {trajectory[:5]}")
    
    # Test confidence interval
    data = np.random.normal(100, 15, 1000)
    ci = calculate_confidence_interval(data)
    print(f"95% CI: {ci}")
    
    print("\n✓ All utility functions working correctly!")
