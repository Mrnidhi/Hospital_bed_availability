"""
Hospital Bed Occupancy Forecasting - Feature Engineering Module

This module handles data preprocessing and feature engineering for hospital bed occupancy forecasting.
Includes data cleaning, merging, and creation of time-series features.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HospitalFeatureEngineer:
    """
    Handles data preprocessing and feature engineering for hospital bed occupancy forecasting.
    """
    
    def __init__(self):
        self.required_columns = [
            'collection_week', 'hospital_name', 'state', 'total_adult_patients_hospitalized_confirmed_covid',
            'total_adult_patients_hospitalized_confirmed_and_suspected_covid', 'total_pediatric_patients_hospitalized_confirmed_covid',
            'total_pediatric_patients_hospitalized_confirmed_and_suspected_covid', 'staffed_adult_icu_bed_occupancy',
            'staffed_icu_adult_patients_confirmed_and_suspected_covid', 'total_staffed_adult_icu_beds',
            'total_icu_beds', 'total_beds', 'all_adult_hospital_beds', 'all_adult_hospital_inpatient_beds',
            'all_pediatric_inpatient_beds', 'all_adult_icu_beds'
        ]
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the raw hospital data.
        
        Args:
            df (pd.DataFrame): Raw hospital data
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        if df.empty:
            logger.warning("Input DataFrame is empty")
            return df
        
        df_clean = df.copy()
        
        # Convert date columns
        if 'collection_week' in df_clean.columns:
            df_clean['collection_week'] = pd.to_datetime(df_clean['collection_week'])
        
        # Convert numeric columns
        numeric_columns = [
            'total_adult_patients_hospitalized_confirmed_covid',
            'total_adult_patients_hospitalized_confirmed_and_suspected_covid',
            'total_pediatric_patients_hospitalized_confirmed_covid',
            'total_pediatric_patients_hospitalized_confirmed_and_suspected_covid',
            'staffed_adult_icu_bed_occupancy',
            'staffed_icu_adult_patients_confirmed_and_suspected_covid',
            'total_staffed_adult_icu_beds',
            'total_icu_beds',
            'total_beds',
            'all_adult_hospital_beds',
            'all_adult_hospital_inpatient_beds',
            'all_pediatric_inpatient_beds',
            'all_adult_icu_beds',
            # Add 7-day average columns
            'total_adult_patients_hospitalized_confirmed_covid_7_day_avg',
            'total_pediatric_patients_hospitalized_confirmed_covid_7_day_avg',
            'total_adult_patients_hospitalized_confirmed_and_suspected_covid_7_day_avg',
            'total_pediatric_patients_hospitalized_confirmed_and_suspected_covid_7_day_avg',
            'staffed_icu_adult_patients_confirmed_covid_7_day_avg',
            'staffed_icu_adult_patients_confirmed_and_suspected_covid_7_day_avg',
            'inpatient_beds_7_day_avg',
            'total_staffed_adult_icu_beds_7_day_avg',
            'total_icu_beds_7_day_avg',
            'all_adult_hospital_inpatient_beds_7_day_avg',
            'inpatient_beds_used_7_day_avg',
            'icu_beds_used_7_day_avg'
        ]
        
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Remove columns with unhashable types (dict/list) before deduplication
        for col in df_clean.columns:
            if df_clean[col].apply(lambda x: isinstance(x, (dict, list))).any():
                df_clean = df_clean.drop(columns=[col])
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Sort by date and hospital
        if 'collection_week' in df_clean.columns and 'hospital_name' in df_clean.columns:
            df_clean = df_clean.sort_values(['hospital_name', 'collection_week'])
        
        logger.info(f"Cleaned data: {len(df_clean)} records")
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with handled missing values
        """
        df_clean = df.copy()
        
        # For numeric columns, fill with 0 or forward fill
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in df_clean.columns:
                # Forward fill within each hospital group
                if 'hospital_name' in df_clean.columns:
                    df_clean[col] = df_clean.groupby('hospital_name')[col].fillna(method='ffill')
                # Fill remaining NaN with 0
                df_clean[col] = df_clean[col].fillna(0)
        
        # For categorical columns, fill with 'Unknown'
        categorical_columns = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna('Unknown')
        
        return df_clean
    
    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create basic features from the cleaned data.
        
        Args:
            df (pd.DataFrame): Cleaned hospital data
            
        Returns:
            pd.DataFrame: Data with basic features added
        """
        if df.empty:
            return df
        
        df_features = df.copy()
        
        # Calculate total patients from 7-day average columns
        # HHS API now provides 7-day averages instead of daily values
        adult_col = 'total_adult_patients_hospitalized_confirmed_covid_7_day_avg'
        pediatric_col = 'total_pediatric_patients_hospitalized_confirmed_covid_7_day_avg'
        
        if all(col in df_features.columns for col in [adult_col, pediatric_col]):
            df_features['total_patients'] = (
                df_features[adult_col].fillna(0) +
                df_features[pediatric_col].fillna(0)
            )
            logger.info(f"Created total_patients from {adult_col} + {pediatric_col}")
        else:
            # Fallback: try alternative column names
            alt_adult_col = 'total_adult_patients_hospitalized_confirmed_and_suspected_covid_7_day_avg'
            alt_pediatric_col = 'total_pediatric_patients_hospitalized_confirmed_and_suspected_covid_7_day_avg'
            
            if all(col in df_features.columns for col in [alt_adult_col, alt_pediatric_col]):
                df_features['total_patients'] = (
                    df_features[alt_adult_col].fillna(0) +
                    df_features[alt_pediatric_col].fillna(0)
                )
                logger.info(f"Created total_patients from {alt_adult_col} + {alt_pediatric_col}")
            else:
                logger.warning("Could not find columns to compute total_patients")
                df_features['total_patients'] = 0
        
        # Calculate ICU patients from 7-day average column
        icu_col = 'staffed_icu_adult_patients_confirmed_covid_7_day_avg'
        if icu_col in df_features.columns:
            df_features['icu_patients'] = df_features[icu_col].fillna(0)
            logger.info(f"Created icu_patients from {icu_col}")
        else:
            # Fallback: try alternative ICU column
            alt_icu_col = 'staffed_icu_adult_patients_confirmed_and_suspected_covid_7_day_avg'
            if alt_icu_col in df_features.columns:
                df_features['icu_patients'] = df_features[alt_icu_col].fillna(0)
                logger.info(f"Created icu_patients from {alt_icu_col}")
            else:
                logger.warning("Could not find columns to compute icu_patients")
                df_features['icu_patients'] = 0
        
        # Calculate bed utilization rates
        # Use 7-day average bed counts
        beds_col = 'inpatient_beds_7_day_avg'
        if all(col in df_features.columns for col in ['total_patients', beds_col]):
            df_features['bed_utilization_rate'] = (
                df_features['total_patients'] / df_features[beds_col].replace(0, 1)
            ).clip(0, 1)
            logger.info(f"Created bed_utilization_rate from total_patients / {beds_col}")
        else:
            logger.warning("Could not compute bed_utilization_rate - missing required columns")
            df_features['bed_utilization_rate'] = 0
        
        # Calculate ICU utilization rate
        icu_beds_col = 'total_staffed_adult_icu_beds_7_day_avg'
        if all(col in df_features.columns for col in ['icu_patients', icu_beds_col]):
            df_features['icu_utilization_rate'] = (
                df_features['icu_patients'] / df_features[icu_beds_col].replace(0, 1)
            ).clip(0, 1)
            logger.info(f"Created icu_utilization_rate from icu_patients / {icu_beds_col}")
        else:
            logger.warning("Could not compute icu_utilization_rate - missing required columns")
            df_features['icu_utilization_rate'] = 0
        
        # Add date-based features
        if 'collection_week' in df_features.columns:
            df_features['year'] = df_features['collection_week'].dt.year
            df_features['month'] = df_features['collection_week'].dt.month
            df_features['day_of_week'] = df_features['collection_week'].dt.dayofweek
            df_features['week_of_year'] = df_features['collection_week'].dt.isocalendar().week
        
        logger.info("Created basic features")
        return df_features
    
    def create_lag_features(self, df: pd.DataFrame, lag_periods: List[int] = [1, 3, 7]) -> pd.DataFrame:
        """
        Create lag features for time-series forecasting.
        
        Args:
            df (pd.DataFrame): Data with basic features
            lag_periods (List[int]): List of lag periods to create
            
        Returns:
            pd.DataFrame: Data with lag features added
        """
        if df.empty or 'hospital_name' not in df.columns:
            return df
        
        df_lags = df.copy()
        
        # Sort by hospital and date
        df_lags = df_lags.sort_values(['hospital_name', 'collection_week'])
        
        # Create lag features for key metrics
        lag_columns = ['total_patients', 'icu_patients', 'bed_utilization_rate', 'icu_utilization_rate']
        
        for col in lag_columns:
            if col in df_lags.columns:
                for lag in lag_periods:
                    lag_col_name = f'{col}_lag_{lag}'
                    df_lags[lag_col_name] = df_lags.groupby('hospital_name')[col].shift(lag)
        
        logger.info(f"Created lag features for periods: {lag_periods}")
        return df_lags
    
    def create_rolling_features(self, df: pd.DataFrame, windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
        """
        Create rolling average features.
        
        Args:
            df (pd.DataFrame): Data with lag features
            windows (List[int]): List of rolling window sizes
            
        Returns:
            pd.DataFrame: Data with rolling features added
        """
        if df.empty or 'hospital_name' not in df.columns:
            return df
        
        df_rolling = df.copy()
        
        # Sort by hospital and date
        df_rolling = df_rolling.sort_values(['hospital_name', 'collection_week'])
        
        # Create rolling features for key metrics
        rolling_columns = ['total_patients', 'icu_patients', 'bed_utilization_rate', 'icu_utilization_rate']
        
        for col in rolling_columns:
            if col in df_rolling.columns:
                for window in windows:
                    # Rolling mean
                    mean_col_name = f'{col}_rolling_mean_{window}'
                    df_rolling[mean_col_name] = df_rolling.groupby('hospital_name')[col].rolling(
                        window=window, min_periods=1
                    ).mean().reset_index(0, drop=True)
                    
                    # Rolling std
                    std_col_name = f'{col}_rolling_std_{window}'
                    df_rolling[std_col_name] = df_rolling.groupby('hospital_name')[col].rolling(
                        window=window, min_periods=1
                    ).std().reset_index(0, drop=True)
        
        logger.info(f"Created rolling features for windows: {windows}")
        return df_rolling
    
    def create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create seasonal and cyclical features.
        
        Args:
            df (pd.DataFrame): Data with rolling features
            
        Returns:
            pd.DataFrame: Data with seasonal features added
        """
        if df.empty or 'collection_week' not in df.columns:
            return df
        
        df_seasonal = df.copy()
        
        # Day of week cyclical encoding
        if 'day_of_week' in df_seasonal.columns:
            df_seasonal['day_of_week_sin'] = np.sin(2 * np.pi * df_seasonal['day_of_week'] / 7)
            df_seasonal['day_of_week_cos'] = np.cos(2 * np.pi * df_seasonal['day_of_week'] / 7)
        
        # Month cyclical encoding
        if 'month' in df_seasonal.columns:
            df_seasonal['month_sin'] = np.sin(2 * np.pi * df_seasonal['month'] / 12)
            df_seasonal['month_cos'] = np.cos(2 * np.pi * df_seasonal['month'] / 12)
        
        # Week of year cyclical encoding
        if 'week_of_year' in df_seasonal.columns:
            df_seasonal['week_of_year_sin'] = np.sin(2 * np.pi * df_seasonal['week_of_year'] / 52)
            df_seasonal['week_of_year_cos'] = np.cos(2 * np.pi * df_seasonal['week_of_year'] / 52)
        
        logger.info("Created seasonal features")
        return df_seasonal
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df (pd.DataFrame): Raw hospital data
            
        Returns:
            pd.DataFrame: Fully engineered dataset ready for forecasting
        """
        logger.info("Starting feature engineering pipeline")
        
        # Step 1: Clean data
        df_clean = self.clean_data(df)
        
        # Step 2: Create basic features
        df_basic = self.create_basic_features(df_clean)
        
        # Step 3: Create lag features
        df_lags = self.create_lag_features(df_basic)
        
        # Step 4: Create rolling features
        df_rolling = self.create_rolling_features(df_lags)
        
        # Step 5: Create seasonal features
        df_final = self.create_seasonal_features(df_rolling)
        
        logger.info(f"Feature engineering complete. Final dataset: {len(df_final)} records, {len(df_final.columns)} features")
        
        return df_final
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of engineered feature columns (excluding raw data columns).
        
        Args:
            df (pd.DataFrame): Engineered dataset
            
        Returns:
            List[str]: List of feature column names
        """
        # Define raw columns to exclude
        raw_columns = [
            'collection_week', 'hospital_name', 'state', 'ingestion_timestamp',
            'total_adult_patients_hospitalized_confirmed_covid',
            'total_adult_patients_hospitalized_confirmed_and_suspected_covid',
            'total_pediatric_patients_hospitalized_confirmed_covid',
            'total_pediatric_patients_hospitalized_confirmed_and_suspected_covid',
            'staffed_adult_icu_bed_occupancy',
            'staffed_icu_adult_patients_confirmed_and_suspected_covid',
            'total_staffed_adult_icu_beds', 'total_icu_beds', 'total_beds',
            'all_adult_hospital_beds', 'all_adult_hospital_inpatient_beds',
            'all_pediatric_inpatient_beds', 'all_adult_icu_beds'
        ]
        
        feature_columns = [col for col in df.columns if col not in raw_columns]
        return feature_columns

def main():
    """
    Example usage of the HospitalFeatureEngineer.
    """
    from streaming import HospitalDataStreamer
    
    # Get some sample data
    streamer = HospitalDataStreamer()
    sample_data = streamer.fetch_hhs_data(limit=100)
    
    if not sample_data.empty:
        # Engineer features
        engineer = HospitalFeatureEngineer()
        engineered_data = engineer.engineer_features(sample_data)
        
        print(f"Original data shape: {sample_data.shape}")
        print(f"Engineered data shape: {engineered_data.shape}")
        print(f"Feature columns: {engineer.get_feature_columns(engineered_data)}")
        
        # Show sample of engineered data
        print("\nSample engineered data:")
        print(engineered_data.head())
    else:
        print("No sample data available")

if __name__ == "__main__":
    main() 