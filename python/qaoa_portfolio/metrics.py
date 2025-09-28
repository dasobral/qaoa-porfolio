"""
Financial metrics for QAOA Portfolio Optimizer (QOPO)

Simple financial calculations for portfolio analysis.
Heavy computations are deferred to Rust.

Author: Daniel Sobral Blanco
License: CC BY-NC-ND 4.0
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class FinancialMetrics:
    """Simple financial calculations and metrics for portfolio analysis."""
    
    @staticmethod
    def simple_return(start_price: float, end_price: float) -> float:
        """Calculate simple return between two prices."""
        if start_price <= 0:
            return 0.0
        return (end_price - start_price) / start_price
    
    @staticmethod
    def log_return(start_price: float, end_price: float) -> float:
        """Calculate logarithmic return between two prices."""
        if start_price <= 0 or end_price <= 0:
            return 0.0
        return np.log(end_price / start_price)
    
    @staticmethod
    def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
        """Annualize a return series."""
        if returns.empty:
            return 0.0
        return (1 + returns.mean()) ** periods_per_year - 1
    
    @staticmethod
    def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
        """Annualize volatility from a return series."""
        if returns.empty:
            return 0.0
        return returns.std() * np.sqrt(periods_per_year)
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio."""
        if returns.empty or returns.std() == 0:
            return 0.0
        
        ann_return = FinancialMetrics.annualized_return(returns, periods_per_year)
        ann_vol = FinancialMetrics.annualized_volatility(returns, periods_per_year)
        
        return (ann_return - risk_free_rate) / ann_vol
    
    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
        """Calculate Sortino ratio (downside deviation version of Sharpe)."""
        if returns.empty:
            return 0.0
        
        ann_return = FinancialMetrics.annualized_return(returns, periods_per_year)
        downside_returns = returns[returns < 0]
        
        if downside_returns.empty:
            return np.inf if ann_return > risk_free_rate else 0.0
        
        downside_vol = downside_returns.std() * np.sqrt(periods_per_year)
        return (ann_return - risk_free_rate) / downside_vol
    
    @staticmethod
    def max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series."""
        if returns.empty:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min()
    
    @staticmethod
    def value_at_risk(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk (VaR)."""
        if returns.empty:
            return 0.0
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def conditional_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        if returns.empty:
            return 0.0
        
        var = FinancialMetrics.value_at_risk(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta coefficient relative to market."""
        if asset_returns.empty or market_returns.empty:
            return 0.0
        
        aligned_data = pd.concat([asset_returns, market_returns], axis=1).dropna()
        if aligned_data.shape[0] < 2:
            return 0.0
        
        asset_aligned = aligned_data.iloc[:, 0]
        market_aligned = aligned_data.iloc[:, 1]
        
        market_var = market_aligned.var()
        if market_var == 0:
            return 0.0
        
        return asset_aligned.cov(market_aligned) / market_var
    
    @staticmethod
    def correlation(returns1: pd.Series, returns2: pd.Series) -> float:
        """Calculate correlation between two return series."""
        if returns1.empty or returns2.empty:
            return 0.0
        
        aligned_data = pd.concat([returns1, returns2], axis=1).dropna()
        if aligned_data.shape[0] < 2:
            return 0.0
        
        return aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])