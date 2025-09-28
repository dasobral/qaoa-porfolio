"""
Configuration settings for QAOA Portfolio Optimizer (QOPO)

This module contains configuration settings, constants, and default values
used throughout the portfolio optimization framework.

Author: Daniel Sobral Blanco
License: CC BY-NC-ND 4.0
"""
import logging 
from typing import Any, Dict, Optional
from pathlib import Path
import json
from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration Management
# ============================================================================

class ConfigManager:
    """Configuration management for the QAOA Portfolio Optimizer."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else Path("config/settings.json")
        self.config = self._load_default_config()
        
        if self.config_path.exists():
            self.load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration optimized for free-tier usage."""
        return {
            "data_sources": {
                "default": "yfinance",  # Free, no API key needed
                "fallback_order": ["yfinance"],  # Add more if API keys are available
                "cache_enabled": True,
                "cache_duration_days": 7,  # Longer cache for free tier
                "free_tier_mode": True
            },
            "optimization": {
                "default_method": "qaoa",
                "max_iterations": 50,  # Reduced for faster testing
                "convergence_threshold": 1e-4,  # Less strict for testing
                "random_seed": 42
            },
            "portfolio": {
                "default_size": 5,  # Smaller default for free tier testing
                "min_weight": 0.0,
                "max_weight": 0.4,  # More diversified for testing
                "risk_free_rate": 0.02,
                "target_return": None
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "show_free_tier_tips": True
            },
            "performance": {
                "enable_monitoring": True,
                "profile_memory": False,
                "parallel_workers": 2,  # Reduced for free tier
                "conservative_rate_limiting": True
            },
            "free_tier": {
                "yahoo_finance": {
                    "enabled": True,
                    "api_key_required": False,
                    "rate_limit_calls_per_minute": 60,
                    "recommended": True
                }
                # Add more data sources here if needed or API keys available
            }
        }
    
    def load_config(self) -> None:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                file_config = json.load(f)
            
            # Merge with default config
            self._deep_update(self.config, file_config)
            logger.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error loading config from {self.config_path}: {e}")
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error saving config to {self.config_path}: {e}")
            raise ConfigurationError(f"Failed to save configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value

    def setup_logging(self) -> None:
        """Setup logging based on configuration."""
        level = self.get('logging.level', 'INFO')
        format_str = self.get('logging.format')
        
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=format_str,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    @staticmethod
    def _deep_update(base_dict: Dict, update_dict: Dict) -> None:
        """Deep update dictionary with another dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                ConfigManager._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

# Global configuration instance

config = ConfigManager()