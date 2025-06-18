"""
Hospital Bed Occupancy Forecasting - Forecasting Module

This module implements time-series forecasting models for hospital bed occupancy prediction.
Includes ARIMA, Exponential Smoothing, XGBoost, and evaluation metrics.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    logging.warning("statsmodels not available. ARIMA and Exponential Smoothing will be disabled.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("xgboost not available. XGBoost model will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HospitalForecaster:
    """
    Implements time-series forecasting models for hospital bed occupancy.
    """
    
    def __init__(self):
        self.models = {}
        self.feature_columns = []
        self.target_columns = ['total_patients', 'icu_patients', 'bed_utilization_rate', 'icu_utilization_rate']
        self.evaluation_metrics = {}
        
    def prepare_data_for_forecasting(self, df: pd.DataFrame, target_col: str, 
                                   hospital_name: Optional[str] = None) -> Tuple[pd.Series, pd.Series]:
        """
        Prepare data for time-series forecasting.
        
        Args:
            df (pd.DataFrame): Engineered dataset
            target_col (str): Target column to forecast
            hospital_name (Optional[str]): Specific hospital to forecast for
            
        Returns:
            Tuple[pd.Series, pd.Series]: Features and target series
        """
        if df.empty:
            return pd.Series(), pd.Series()
        
        # Filter by hospital if specified
        if hospital_name:
            df_filtered = df[df['hospital_name'] == hospital_name].copy()
        else:
            df_filtered = df.copy()
        
        if df_filtered.empty:
            logger.warning(f"No data found for hospital: {hospital_name}")
            return pd.Series(), pd.Series()
        
        # Sort by date
        if 'collection_week' in df_filtered.columns:
            df_filtered = df_filtered.sort_values('collection_week')
        
        # Get target series
        if target_col not in df_filtered.columns:
            logger.error(f"Target column {target_col} not found in data")
            return pd.Series(), pd.Series()
        
        target_series = df_filtered[target_col].fillna(method='ffill').fillna(0)
        
        # Get feature columns (exclude target and metadata columns)
        exclude_cols = ['collection_week', 'hospital_name', 'state', 'ingestion_timestamp', target_col]
        feature_cols = [col for col in df_filtered.columns if col not in exclude_cols]
        
        # Create feature matrix
        feature_matrix = df_filtered[feature_cols].fillna(0)
        
        return feature_matrix, target_series
    
    def train_arima_model(self, target_series: pd.Series, order: Tuple[int, int, int] = (1, 1, 1)) -> Any:
        """
        Train ARIMA model for time-series forecasting.
        
        Args:
            target_series (pd.Series): Target time series
            order (Tuple[int, int, int]): ARIMA order (p, d, q)
            
        Returns:
            Any: Trained ARIMA model
        """
        if not ARIMA_AVAILABLE:
            logger.error("ARIMA not available. Install statsmodels.")
            return None
        
        try:
            # Remove any remaining NaN values
            clean_series = target_series.dropna()
            
            if len(clean_series) < 10:
                logger.warning("Insufficient data for ARIMA model")
                return None
            
            model = ARIMA(clean_series, order=order)
            fitted_model = model.fit()
            
            logger.info(f"ARIMA model trained successfully with order {order}")
            return fitted_model
            
        except Exception as e:
            logger.error(f"Error training ARIMA model: {e}")
            return None
    
    def train_exponential_smoothing_model(self, target_series: pd.Series, 
                                        seasonal_periods: int = 7) -> Any:
        """
        Train Exponential Smoothing model for time-series forecasting.
        
        Args:
            target_series (pd.Series): Target time series
            seasonal_periods (int): Number of seasonal periods
            
        Returns:
            Any: Trained Exponential Smoothing model
        """
        if not ARIMA_AVAILABLE:
            logger.error("Exponential Smoothing not available. Install statsmodels.")
            return None
        
        try:
            # Remove any remaining NaN values
            clean_series = target_series.dropna()
            
            if len(clean_series) < seasonal_periods * 2:
                logger.warning("Insufficient data for Exponential Smoothing model")
                return None
            
            model = ExponentialSmoothing(
                clean_series, 
                seasonal_periods=seasonal_periods,
                trend='add',
                seasonal='add'
            )
            fitted_model = model.fit()
            
            logger.info(f"Exponential Smoothing model trained successfully")
            return fitted_model
            
        except Exception as e:
            logger.error(f"Error training Exponential Smoothing model: {e}")
            return None
    
    def train_xgboost_model(self, feature_matrix: pd.DataFrame, target_series: pd.Series,
                           test_size: float = 0.2) -> Any:
        """
        Train XGBoost model for time-series forecasting.
        
        Args:
            feature_matrix (pd.DataFrame): Feature matrix
            target_series (pd.Series): Target series
            test_size (float): Proportion of data for testing
            
        Returns:
            Any: Trained XGBoost model
        """
        if not XGBOOST_AVAILABLE:
            logger.error("XGBoost not available. Install xgboost.")
            return None
        
        try:
            # Align features and target
            common_length = min(len(feature_matrix), len(target_series))
            X = feature_matrix.iloc[:common_length]
            y = target_series.iloc[:common_length]
            
            # Remove rows with NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            if len(X) < 10:
                logger.warning("Insufficient data for XGBoost model")
                return None
            
            # Split data (time-series aware)
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Train model
            model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                early_stopping_rounds=10
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            
            logger.info(f"XGBoost model trained successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")
            return None
    
    def forecast_arima(self, model: Any, steps: int = 7) -> np.ndarray:
        """
        Make forecasts using ARIMA model.
        
        Args:
            model: Trained ARIMA model
            steps (int): Number of steps to forecast
            
        Returns:
            np.ndarray: Forecasted values
        """
        if model is None:
            return np.array([])
        
        try:
            forecast = model.forecast(steps=steps)
            return forecast.values if hasattr(forecast, 'values') else forecast
        except Exception as e:
            logger.error(f"Error making ARIMA forecast: {e}")
            return np.array([])
    
    def forecast_exponential_smoothing(self, model: Any, steps: int = 7) -> np.ndarray:
        """
        Make forecasts using Exponential Smoothing model.
        
        Args:
            model: Trained Exponential Smoothing model
            steps (int): Number of steps to forecast
            
        Returns:
            np.ndarray: Forecasted values
        """
        if model is None:
            return np.array([])
        
        try:
            forecast = model.forecast(steps)
            return forecast.values if hasattr(forecast, 'values') else forecast
        except Exception as e:
            logger.error(f"Error making Exponential Smoothing forecast: {e}")
            return np.array([])
    
    def forecast_xgboost(self, model: Any, feature_matrix: pd.DataFrame, steps: int = 7) -> np.ndarray:
        """
        Make forecasts using XGBoost model.
        
        Args:
            model: Trained XGBoost model
            feature_matrix (pd.DataFrame): Feature matrix for forecasting
            steps (int): Number of steps to forecast
            
        Returns:
            np.ndarray: Forecasted values
        """
        if model is None:
            return np.array([])
        
        try:
            # For XGBoost, we need to create future feature values
            # This is a simplified approach - in practice, you'd need more sophisticated feature generation
            if len(feature_matrix) >= steps:
                future_features = feature_matrix.iloc[-steps:].copy()
            else:
                # If not enough data, repeat the last row
                last_row = feature_matrix.iloc[-1:].copy()
                future_features = pd.concat([last_row] * steps, ignore_index=True)
            
            forecast = model.predict(future_features)
            return forecast
            
        except Exception as e:
            logger.error(f"Error making XGBoost forecast: {e}")
            return np.array([])
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Dict[str, float]:
        """
        Evaluate model performance using multiple metrics.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            model_name (str): Name of the model
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return {}
        
        # Ensure same length
        min_length = min(len(y_true), len(y_pred))
        y_true = y_true[:min_length]
        y_pred = y_pred[:min_length]
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,  # Add small epsilon to avoid division by zero
            'r2': 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        }
        
        logger.info(f"{model_name} - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
        
        return metrics
    
    def train_all_models(self, df: pd.DataFrame, target_col: str, 
                        hospital_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Train all available forecasting models.
        
        Args:
            df (pd.DataFrame): Engineered dataset
            target_col (str): Target column to forecast
            hospital_name (Optional[str]): Specific hospital to forecast for
            
        Returns:
            Dict[str, Any]: Dictionary of trained models
        """
        logger.info(f"Training models for target: {target_col}, hospital: {hospital_name}")
        
        # Prepare data
        feature_matrix, target_series = self.prepare_data_for_forecasting(df, target_col, hospital_name)
        
        if target_series.empty:
            logger.error("No data available for training")
            return {}
        
        models = {}
        
        # Train ARIMA model
        if ARIMA_AVAILABLE:
            arima_model = self.train_arima_model(target_series)
            if arima_model is not None:
                models['arima'] = arima_model
        
        # Train Exponential Smoothing model
        if ARIMA_AVAILABLE:
            es_model = self.train_exponential_smoothing_model(target_series)
            if es_model is not None:
                models['exponential_smoothing'] = es_model
        
        # Train XGBoost model
        if XGBOOST_AVAILABLE and not feature_matrix.empty:
            xgb_model = self.train_xgboost_model(feature_matrix, target_series)
            if xgb_model is not None:
                models['xgboost'] = xgb_model
        
        self.models = models
        self.feature_columns = list(feature_matrix.columns) if not feature_matrix.empty else []
        
        logger.info(f"Trained {len(models)} models: {list(models.keys())}")
        return models
    
    def make_forecasts(self, df: pd.DataFrame, target_col: str, steps: int = 7,
                      hospital_name: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Make forecasts using all trained models.
        
        Args:
            df (pd.DataFrame): Engineered dataset
            target_col (str): Target column to forecast
            steps (int): Number of steps to forecast
            hospital_name (Optional[str]): Specific hospital to forecast for
            
        Returns:
            Dict[str, np.ndarray]: Dictionary of forecasts for each model
        """
        if not self.models:
            logger.error("No models trained. Call train_all_models first.")
            return {}
        
        # Prepare data for forecasting
        feature_matrix, target_series = self.prepare_data_for_forecasting(df, target_col, hospital_name)
        
        forecasts = {}
        
        # ARIMA forecast
        if 'arima' in self.models:
            forecasts['arima'] = self.forecast_arima(self.models['arima'], steps)
        
        # Exponential Smoothing forecast
        if 'exponential_smoothing' in self.models:
            forecasts['exponential_smoothing'] = self.forecast_exponential_smoothing(
                self.models['exponential_smoothing'], steps
            )
        
        # XGBoost forecast
        if 'xgboost' in self.models and not feature_matrix.empty:
            forecasts['xgboost'] = self.forecast_xgboost(self.models['xgboost'], feature_matrix, steps)
        
        logger.info(f"Generated forecasts for {len(forecasts)} models")
        return forecasts
    
    def evaluate_all_models(self, df: pd.DataFrame, target_col: str,
                          hospital_name: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models using time-series cross-validation.
        
        Args:
            df (pd.DataFrame): Engineered dataset
            target_col (str): Target column to evaluate
            hospital_name (Optional[str]): Specific hospital to evaluate
            
        Returns:
            Dict[str, Dict[str, float]]: Evaluation metrics for each model
        """
        if not self.models:
            logger.error("No models trained. Call train_all_models first.")
            return {}
        
        # Prepare data
        feature_matrix, target_series = self.prepare_data_for_forecasting(df, target_col, hospital_name)
        
        if target_series.empty:
            logger.error("No data available for evaluation")
            return {}
        
        evaluation_results = {}
        
        # Simple holdout evaluation (last 20% of data)
        split_idx = int(len(target_series) * 0.8)
        y_train = target_series.iloc[:split_idx]
        y_test = target_series.iloc[split_idx:]
        
        if len(y_test) < 5:
            logger.warning("Insufficient test data for evaluation")
            return {}
        
        # Evaluate each model
        for model_name, model in self.models.items():
            try:
                if model_name == 'arima':
                    # Retrain ARIMA on training data
                    train_model = self.train_arima_model(y_train)
                    if train_model is not None:
                        y_pred = self.forecast_arima(train_model, len(y_test))
                        evaluation_results[model_name] = self.evaluate_model(y_test.values, y_pred, model_name)
                
                elif model_name == 'exponential_smoothing':
                    # Retrain Exponential Smoothing on training data
                    train_model = self.train_exponential_smoothing_model(y_train)
                    if train_model is not None:
                        y_pred = self.forecast_exponential_smoothing(train_model, len(y_test))
                        evaluation_results[model_name] = self.evaluate_model(y_test.values, y_pred, model_name)
                
                elif model_name == 'xgboost':
                    # For XGBoost, we need features for the test period
                    if not feature_matrix.empty:
                        X_train = feature_matrix.iloc[:split_idx]
                        X_test = feature_matrix.iloc[split_idx:]
                        train_model = self.train_xgboost_model(X_train, y_train)
                        if train_model is not None:
                            y_pred = self.forecast_xgboost(train_model, X_test, len(y_test))
                            evaluation_results[model_name] = self.evaluate_model(y_test.values, y_pred, model_name)
            
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
        
        self.evaluation_metrics = evaluation_results
        return evaluation_results

def main():
    """
    Example usage of the HospitalForecaster.
    """
    from streaming import HospitalDataStreamer
    from features import HospitalFeatureEngineer
    
    # Get sample data
    streamer = HospitalDataStreamer()
    sample_data = streamer.fetch_hhs_data(limit=200)
    
    if not sample_data.empty:
        # Engineer features
        engineer = HospitalFeatureEngineer()
        engineered_data = engineer.engineer_features(sample_data)
        
        # Initialize forecaster
        forecaster = HospitalForecaster()
        
        # Train models for total patients
        models = forecaster.train_all_models(engineered_data, 'total_patients')
        
        if models:
            # Make forecasts
            forecasts = forecaster.make_forecasts(engineered_data, 'total_patients', steps=7)
            
            print("Forecasts for next 7 days:")
            for model_name, forecast in forecasts.items():
                print(f"{model_name}: {forecast}")
            
            # Evaluate models
            evaluation = forecaster.evaluate_all_models(engineered_data, 'total_patients')
            
            print("\nModel Evaluation:")
            for model_name, metrics in evaluation.items():
                print(f"{model_name}: {metrics}")
        else:
            print("No models were successfully trained")
    else:
        print("No sample data available")

if __name__ == "__main__":
    main() 