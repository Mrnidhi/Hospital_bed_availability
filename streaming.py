"""
Hospital Bed Occupancy Forecasting - Data Ingestion Module

This module simulates real-time data streaming from public hospital APIs:
- HHS Protect (Hospital Weekly Capacity)
- CMS Hospital General Information

Author: AI Assistant
Date: 2024
"""

import requests
import pandas as pd
import time
import json
from datetime import datetime, timedelta
from typing import Generator, Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HospitalDataStreamer:
    """
    Simulates real-time data streaming from hospital APIs.
    
    Fetches data from:
    - HHS Protect: Hospital weekly capacity data
    - CMS: Hospital general information
    """
    
    def __init__(self):
        self.hhs_url = "https://healthdata.gov/resource/anag-cw7u.json"
        self.cms_url = "https://data.cms.gov/provider-data/api/views/xubh-q36u/rows.json?accessType=DOWNLOAD"
        self.data_buffer = []
        self.last_fetch_time = None
        
    def fetch_hhs_data(self, limit: int = 100) -> pd.DataFrame:
        """
        Fetch hospital capacity data from HHS Protect API.
        
        Args:
            limit (int): Number of records to fetch
            
        Returns:
            pd.DataFrame: Hospital capacity data
        """
        try:
            params = {
                '$limit': limit,
                '$order': 'collection_week DESC'
            }
            
            response = requests.get(self.hhs_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data)
            
            # Convert date columns
            if 'collection_week' in df.columns:
                df['collection_week'] = pd.to_datetime(df['collection_week'])
            
            logger.info(f"Fetched {len(df)} records from HHS Protect API")
            return df
            
        except requests.RequestException as e:
            logger.error(f"Error fetching HHS data: {e}")
            return pd.DataFrame()
    
    def fetch_cms_data(self) -> pd.DataFrame:
        """
        Fetch hospital general information from CMS API.
        
        Returns:
            pd.DataFrame: Hospital general information
        """
        try:
            response = requests.get(self.cms_url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data['data'])
            
            # Set column names from meta
            if 'meta' in data and 'view' in data['meta'] and 'columns' in data['meta']['view']:
                columns = [col['name'] for col in data['meta']['view']['columns']]
                df.columns = columns
            
            logger.info(f"Fetched {len(df)} records from CMS API")
            return df
            
        except requests.RequestException as e:
            logger.error(f"Error fetching CMS data: {e}")
            return pd.DataFrame()
    
    def merge_datasets(self, hhs_df: pd.DataFrame, cms_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge HHS and CMS datasets on hospital identifiers.
        
        Args:
            hhs_df (pd.DataFrame): HHS hospital capacity data
            cms_df (pd.DataFrame): CMS hospital information
            
        Returns:
            pd.DataFrame: Merged dataset
        """
        if hhs_df.empty or cms_df.empty:
            logger.warning("One or both datasets are empty, returning empty DataFrame")
            return pd.DataFrame()
        
        # Try to merge on hospital name or ID
        # This is a simplified merge - in practice, you'd need more sophisticated matching
        merged_df = hhs_df.copy()
        
        # Add timestamp for tracking
        merged_df['ingestion_timestamp'] = datetime.now()
        
        logger.info(f"Merged dataset contains {len(merged_df)} records")
        return merged_df
    
    def stream_data(self, interval_seconds: int = 30, max_iterations: int = 10) -> Generator[pd.DataFrame, None, None]:
        """
        Simulate real-time data streaming by fetching data at regular intervals.
        
        Args:
            interval_seconds (int): Time between data fetches
            max_iterations (int): Maximum number of iterations (None for infinite)
            
        Yields:
            pd.DataFrame: New data chunk
        """
        iteration = 0
        
        while max_iterations is None or iteration < max_iterations:
            try:
                logger.info(f"Streaming iteration {iteration + 1}")
                
                # Fetch data from both APIs
                hhs_data = self.fetch_hhs_data(limit=50)  # Fetch smaller chunks for streaming
                cms_data = self.fetch_cms_data()
                
                # Merge datasets
                merged_data = self.merge_datasets(hhs_data, cms_data)
                
                if not merged_data.empty:
                    self.data_buffer.append(merged_data)
                    self.last_fetch_time = datetime.now()
                    yield merged_data
                else:
                    logger.warning("No data received in this iteration")
                
                iteration += 1
                
                # Wait before next fetch
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in streaming iteration {iteration + 1}: {e}")
                time.sleep(interval_seconds)
                iteration += 1
    
    def get_latest_data(self) -> pd.DataFrame:
        """
        Get the most recent data from the buffer.
        
        Returns:
            pd.DataFrame: Latest data chunk
        """
        if self.data_buffer:
            return self.data_buffer[-1]
        return pd.DataFrame()
    
    def get_all_buffered_data(self) -> pd.DataFrame:
        """
        Get all data from the buffer as a single DataFrame.
        
        Returns:
            pd.DataFrame: All buffered data
        """
        if self.data_buffer:
            return pd.concat(self.data_buffer, ignore_index=True)
        return pd.DataFrame()

def main():
    """
    Example usage of the HospitalDataStreamer.
    """
    streamer = HospitalDataStreamer()
    
    print("Starting hospital data streaming simulation...")
    print("Press Ctrl+C to stop")
    
    try:
        for i, data_chunk in enumerate(streamer.stream_data(interval_seconds=30, max_iterations=5)):
            print(f"Received data chunk {i+1}: {len(data_chunk)} records")
            print(f"Columns: {list(data_chunk.columns)}")
            print(f"Timestamp: {datetime.now()}")
            print("-" * 50)
            
    except KeyboardInterrupt:
        print("\nStreaming stopped by user")
    
    # Show final buffered data
    final_data = streamer.get_all_buffered_data()
    print(f"\nTotal records buffered: {len(final_data)}")

if __name__ == "__main__":
    main() 