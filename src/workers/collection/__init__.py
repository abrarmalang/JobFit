"""
Data Collection Module

Handles data collection from various job sources (Adzuna API, RemoteOK API, web scraping, etc.)
"""

from .collector_adzuna import AdzunaCollector
from .collector_remoteok import RemoteOKCollector

__all__ = ["AdzunaCollector", "RemoteOKCollector"]
