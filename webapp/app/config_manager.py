"""
Configuration Manager for LLM Settings

Handles saving and loading LLM API keys and settings from JSON file.
Keeps config in memory for fast access.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """Manages LLM configuration with in-memory caching."""

    def __init__(self, config_dir: Optional[Path] = None):
        if config_dir is None:
            config_dir = Path.cwd() / "config"

        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        self.config_file = self.config_dir / "llm_config.json"

        # In-memory cache
        self._config_cache = None

    def save_config(self, config: Dict[str, Any]) -> bool:
        """
        Save LLM configuration to JSON file and update cache.

        Args:
            config: Configuration dict with provider, api_key, model, etc.

        Returns:
            True if saved successfully
        """
        try:
            # Save to file
            self.config_file.write_text(json.dumps(config, indent=2))

            # Update cache
            self._config_cache = config.copy()

            # Also set as environment variable for current process
            self._set_env_var(config)

            return True

        except Exception as e:
            print(f"Error saving config: {e}")
            return False

    def load_config(self) -> Dict[str, Any]:
        """
        Load LLM configuration from cache or file.

        Returns:
            Configuration dict
        """
        # Return from cache if available
        if self._config_cache is not None:
            return self._config_cache.copy()

        # Load from file
        if not self.config_file.exists():
            self._config_cache = self._get_default_config()
            return self._config_cache.copy()

        try:
            config = json.loads(self.config_file.read_text())
            self._config_cache = config
            return config.copy()

        except Exception as e:
            print(f"Error loading config: {e}")
            self._config_cache = self._get_default_config()
            return self._config_cache.copy()

    def _set_env_var(self, config: Dict[str, Any]):
        """Set environment variable for current process."""
        provider = config.get('provider')
        api_key = config.get('api_key')

        if not provider or provider == 'none' or not api_key:
            return

        # Map provider to environment variable name
        env_var_map = {
            'groq': 'GROQ_API_KEY',
            'gemini': 'GEMINI_API_KEY',
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'huggingface': 'HF_API_KEY'
        }

        env_var = env_var_map.get(provider)
        if env_var:
            os.environ[env_var] = api_key

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "provider": "none",
            "api_key": "",
            "model": "",
            "temperature": 0.1,
            "max_tokens": 1000,
            "enable_fallback": True
        }


# Singleton instance
_config_manager = None


def get_config_manager() -> ConfigManager:
    """Get singleton ConfigManager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
