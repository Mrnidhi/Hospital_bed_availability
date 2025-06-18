"""
Hospital Bed Occupancy Forecasting - Dashboard Module

This module creates a Streamlit dashboard for visualizing hospital bed occupancy forecasts.
Includes real-time data updates, interactive charts, and forecast comparisons.

Author: AI Assistant
Date: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import threading
from typing import Dict, List, Optional
import logging

# Import our modules
from streaming import HospitalDataStreamer
from features import HospitalFeatureEngineer
from forecast import HospitalForecaster

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HospitalDashboard:
    """
    Streamlit dashboard for hospital bed occupancy forecasting.
    """
    
    def __init__(self):
        self.streamer = HospitalDataStreamer()
        self.engineer = HospitalFeatureEngineer()
        self.forecaster = HospitalForecaster()
        self.data_buffer = []
        self.last_update = None
        
    def setup_page(self):
        """
        Setup Streamlit page configuration.
        """
        st.set_page_config(
            page_title="Hospital Bed Occupancy Forecasting",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🏥 Hospital Bed Occupancy Forecasting Dashboard")
        st.markdown("---")
        
    def create_sidebar(self):
        """
        Create sidebar controls.
        """
        st.sidebar.header("📊 Dashboard Controls")
        
        # Data source selection
        st.sidebar.subheader("Data Source")
        data_source = st.sidebar.selectbox(
            "Select Data Source",
            ["Live Stream", "Sample Data"],
            help="Choose between live data streaming or sample data"
        )
        
        # Hospital selection
        st.sidebar.subheader("Hospital Selection")
        hospital_filter = st.sidebar.selectbox(
            "Select Hospital",
            ["All Hospitals", "Select Specific Hospital"],
            help="Choose to view all hospitals or a specific one"
        )
        
        # Target metric selection
        st.sidebar.subheader("Forecast Target")
        target_metric = st.sidebar.selectbox(
            "Select Target Metric",
            ["total_patients", "icu_patients", "bed_utilization_rate", "icu_utilization_rate"],
            help="Choose which metric to forecast"
        )
        
        # Forecast horizon
        st.sidebar.subheader("Forecast Settings")
        forecast_horizon = st.sidebar.slider(
            "Forecast Horizon (days)",
            min_value=1,
            max_value=30,
            value=7,
            help="Number of days to forecast"
        )
        
        # Model selection
        st.sidebar.subheader("Models")
        models_to_show = st.sidebar.multiselect(
            "Select Models to Display",
            ["ARIMA", "Exponential Smoothing", "XGBoost"],
            default=["ARIMA", "XGBoost"],
            help="Choose which forecasting models to display"
        )
        
        # Auto-refresh settings
        st.sidebar.subheader("Auto-refresh")
        auto_refresh = st.sidebar.checkbox(
            "Enable Auto-refresh",
            value=True,
            help="Automatically refresh data every 30 seconds"
        )
        
        refresh_interval = st.sidebar.slider(
            "Refresh Interval (seconds)",
            min_value=10,
            max_value=60,
            value=30,
            help="How often to refresh the data"
        )
        
        return {
            'data_source': data_source,
            'hospital_filter': hospital_filter,
            'target_metric': target_metric,
            'forecast_horizon': forecast_horizon,
            'models_to_show': models_to_show,
            'auto_refresh': auto_refresh,
            'refresh_interval': refresh_interval
        }
    
    def get_data(self, data_source: str) -> pd.DataFrame:
        """
        Get data based on the selected source.
        
        Args:
            data_source (str): Data source selection
            
        Returns:
            pd.DataFrame: Processed data
        """
        if data_source == "Live Stream":
            # Get latest data from streamer
            latest_data = self.streamer.get_latest_data()
            if not latest_data.empty:
                self.data_buffer.append(latest_data)
                # Keep only last 100 chunks to prevent memory issues
                if len(self.data_buffer) > 100:
                    self.data_buffer = self.data_buffer[-100:]
            
            # Combine all buffered data
            if self.data_buffer:
                combined_data = pd.concat(self.data_buffer, ignore_index=True)
                return combined_data
            else:
                # Fallback to sample data if no live data
                return self.streamer.fetch_hhs_data(limit=200)
        else:
            # Use sample data
            return self.streamer.fetch_hhs_data(limit=200)
    
    def create_summary_metrics(self, df: pd.DataFrame, target_metric: str):
        """
        Create summary metrics cards.
        
        Args:
            df (pd.DataFrame): Processed data
            target_metric (str): Target metric to display
        """
        if df.empty:
            st.warning("No data available for summary metrics")
            return
        
        # Calculate summary statistics
        if target_metric in df.columns:
            current_value = df[target_metric].iloc[-1] if len(df) > 0 else 0
            avg_value = df[target_metric].mean()
            max_value = df[target_metric].max()
            min_value = df[target_metric].min()
            
            # Create metric columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Current Value",
                    value=f"{current_value:.1f}",
                    delta=f"{current_value - avg_value:.1f}"
                )
            
            with col2:
                st.metric(
                    label="Average",
                    value=f"{avg_value:.1f}",
                    delta=None
                )
            
            with col3:
                st.metric(
                    label="Maximum",
                    value=f"{max_value:.1f}",
                    delta=None
                )
            
            with col4:
                st.metric(
                    label="Minimum",
                    value=f"{min_value:.1f}",
                    delta=None
                )
    
    def create_time_series_plot(self, df: pd.DataFrame, target_metric: str, 
                               forecasts: Dict[str, np.ndarray], models_to_show: List[str]):
        """
        Create time series plot with historical data and forecasts.
        
        Args:
            df (pd.DataFrame): Historical data
            target_metric (str): Target metric
            forecasts (Dict[str, np.ndarray]): Forecast data
            models_to_show (List[str]): Models to display
        """
        if df.empty:
            st.warning("No data available for time series plot")
            return
        
        # Prepare historical data
        if 'collection_week' in df.columns and target_metric in df.columns:
            # Get the most recent data for a single hospital or aggregate
            if 'hospital_name' in df.columns:
                # Use the hospital with most data
                hospital_counts = df['hospital_name'].value_counts()
                if not hospital_counts.empty:
                    main_hospital = hospital_counts.index[0]
                    plot_data = df[df['hospital_name'] == main_hospital].copy()
                else:
                    plot_data = df.copy()
            else:
                plot_data = df.copy()
            
            plot_data = plot_data.sort_values('collection_week')
            
            # Create the plot
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=plot_data['collection_week'],
                y=plot_data[target_metric],
                mode='lines+markers',
                name='Historical Data',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))
            
            # Add forecasts
            if forecasts:
                # Generate future dates
                last_date = plot_data['collection_week'].iloc[-1]
                future_dates = pd.date_range(
                    start=last_date + timedelta(days=1),
                    periods=len(list(forecasts.values())[0]),
                    freq='D'
                )
                
                colors = ['red', 'green', 'orange', 'purple']
                for i, (model_name, forecast_values) in enumerate(forecasts.items()):
                    if model_name.lower() in [m.lower() for m in models_to_show]:
                        fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=forecast_values,
                            mode='lines+markers',
                            name=f'{model_name} Forecast',
                            line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                            marker=dict(size=6)
                        ))
            
            # Update layout
            fig.update_layout(
                title=f'{target_metric.replace("_", " ").title()} - Historical Data and Forecasts',
                xaxis_title='Date',
                yaxis_title=target_metric.replace("_", " ").title(),
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def create_forecast_comparison(self, forecasts: Dict[str, np.ndarray], models_to_show: List[str]):
        """
        Create forecast comparison chart.
        
        Args:
            forecasts (Dict[str, np.ndarray]): Forecast data
            models_to_show (List[str]): Models to display
        """
        if not forecasts:
            st.warning("No forecasts available for comparison")
            return
        
        # Create comparison DataFrame
        forecast_data = []
        for model_name, forecast_values in forecasts.items():
            if model_name.lower() in [m.lower() for m in models_to_show]:
                for i, value in enumerate(forecast_values):
                    forecast_data.append({
                        'Day': i + 1,
                        'Model': model_name,
                        'Forecast': value
                    })
        
        if forecast_data:
            forecast_df = pd.DataFrame(forecast_data)
            
            # Create comparison plot
            fig = px.line(
                forecast_df,
                x='Day',
                y='Forecast',
                color='Model',
                title='Forecast Comparison by Model',
                markers=True
            )
            
            fig.update_layout(
                xaxis_title='Days Ahead',
                yaxis_title='Forecasted Value',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def create_regional_heatmap(self, df: pd.DataFrame, target_metric: str):
        """
        Create regional heatmap of hospital metrics.
        
        Args:
            df (pd.DataFrame): Processed data
            target_metric (str): Target metric
        """
        if df.empty or 'state' not in df.columns or target_metric not in df.columns:
            st.warning("No regional data available for heatmap")
            return
        
        # Aggregate by state
        state_data = df.groupby('state')[target_metric].agg(['mean', 'max', 'min']).reset_index()
        
        if not state_data.empty:
            # Create heatmap
            fig = px.choropleth(
                state_data,
                locations='state',
                locationmode='USA-states',
                color='mean',
                hover_name='state',
                hover_data=['max', 'min'],
                title=f'Regional {target_metric.replace("_", " ").title()} Heatmap',
                color_continuous_scale='Reds'
            )
            
            fig.update_layout(
                geo=dict(
                    scope='usa',
                    projection=dict(type='albers usa'),
                    showlakes=True,
                    lakecolor='rgb(255, 255, 255)'
                ),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def create_model_evaluation_table(self, evaluation_metrics: Dict[str, Dict[str, float]]):
        """
        Create model evaluation metrics table.
        
        Args:
            evaluation_metrics (Dict[str, Dict[str, float]]): Model evaluation results
        """
        if not evaluation_metrics:
            st.warning("No model evaluation metrics available")
            return
        
        # Convert to DataFrame for display
        eval_data = []
        for model_name, metrics in evaluation_metrics.items():
            row = {'Model': model_name}
            row.update(metrics)
            eval_data.append(row)
        
        eval_df = pd.DataFrame(eval_data)
        
        st.subheader("Model Performance Metrics")
        st.dataframe(eval_df, use_container_width=True)
    
    def run_dashboard(self):
        """
        Main dashboard execution function.
        """
        self.setup_page()
        
        # Create sidebar and get settings
        settings = self.create_sidebar()
        
        # Main content area
        st.header("📈 Real-time Hospital Bed Occupancy Analysis")
        
        # Get data
        with st.spinner("Fetching data..."):
            raw_data = self.get_data(settings['data_source'])
        
        if raw_data.empty:
            st.error("No data available. Please check your data source.")
            return
        
        # Engineer features
        with st.spinner("Processing data..."):
            engineered_data = self.engineer.engineer_features(raw_data)
        
        if engineered_data.empty:
            st.error("Failed to process data. Please check data quality.")
            return
        
        # Display summary metrics
        st.subheader("📊 Current Metrics")
        self.create_summary_metrics(engineered_data, settings['target_metric'])
        
        # Train models and generate forecasts
        if st.button("🔄 Generate Forecasts", type="primary"):
            with st.spinner("Training models and generating forecasts..."):
                # Train models
                models = self.forecaster.train_all_models(
                    engineered_data, 
                    settings['target_metric']
                )
                
                if models:
                    # Generate forecasts
                    forecasts = self.forecaster.make_forecasts(
                        engineered_data,
                        settings['target_metric'],
                        settings['forecast_horizon']
                    )
                    
                    # Evaluate models
                    evaluation = self.forecaster.evaluate_all_models(
                        engineered_data,
                        settings['target_metric']
                    )
                    
                    # Store results in session state
                    st.session_state['forecasts'] = forecasts
                    st.session_state['evaluation'] = evaluation
                    st.session_state['models'] = models
                    
                    st.success("Forecasts generated successfully!")
                else:
                    st.error("Failed to train models. Please check data quality.")
        
        # Display forecasts if available
        if 'forecasts' in st.session_state and st.session_state['forecasts']:
            st.subheader("📈 Time Series Analysis")
            self.create_time_series_plot(
                engineered_data,
                settings['target_metric'],
                st.session_state['forecasts'],
                settings['models_to_show']
            )
            
            st.subheader("🔍 Forecast Comparison")
            self.create_forecast_comparison(
                st.session_state['forecasts'],
                settings['models_to_show']
            )
            
            # Display model evaluation
            if 'evaluation' in st.session_state:
                self.create_model_evaluation_table(st.session_state['evaluation'])
        
        # Regional analysis
        st.subheader("🗺️ Regional Analysis")
        self.create_regional_heatmap(engineered_data, settings['target_metric'])
        
        # Data info
        with st.expander("📋 Data Information"):
            st.write(f"**Total Records:** {len(engineered_data)}")
            st.write(f"**Date Range:** {engineered_data['collection_week'].min()} to {engineered_data['collection_week'].max()}")
            st.write(f"**Hospitals:** {engineered_data['hospital_name'].nunique()}")
            st.write(f"**States:** {engineered_data['state'].nunique()}")
            
            st.write("**Sample Data:**")
            st.dataframe(engineered_data.head(), use_container_width=True)
        
        # Auto-refresh logic
        if settings['auto_refresh']:
            st.info(f"🔄 Auto-refresh enabled. Next update in {settings['refresh_interval']} seconds.")
            time.sleep(settings['refresh_interval'])
            st.experimental_rerun()

def main():
    """
    Main function to run the dashboard.
    """
    dashboard = HospitalDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main() 