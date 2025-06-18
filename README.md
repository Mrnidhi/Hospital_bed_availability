# 🏥 Hospital Bed Occupancy Forecasting System

A real-time hospital bed occupancy forecasting system that simulates data ingestion from public U.S. government APIs and applies advanced time-series forecasting techniques.

## 📋 Project Overview

This system provides:
- **Real-time data ingestion** from HHS Protect and CMS APIs
- **Advanced feature engineering** with lag features, rolling averages, and seasonal patterns
- **Multiple forecasting models** (ARIMA, Exponential Smoothing, XGBoost)
- **Interactive dashboard** with live updates and visualizations
- **Model evaluation** with comprehensive metrics

## 🏗️ Architecture

```
Hospital project/
├── streaming.py      # Data ingestion & streaming simulation
├── features.py       # Preprocessing & feature engineering  
├── forecast.py       # Forecasting models & evaluation
├── dashboard.py      # Streamlit dashboard
├── requirements.txt  # Dependencies
└── README.md         # Documentation
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
streamlit run dashboard.py
```

### 3. Test Individual Modules

```bash
# Test data streaming
python streaming.py

# Test feature engineering
python features.py

# Test forecasting
python forecast.py
```

## 📊 Data Sources

### HHS Protect API
- **Endpoint**: `https://healthdata.gov/resource/anag-cw7u.json`
- **Data**: Hospital weekly capacity and COVID-19 patient data
- **Fields**: Total beds, ICU beds, patient counts, utilization rates

### CMS Hospital Information
- **Endpoint**: `https://data.cms.gov/provider-data/api/views/xubh-q36u/rows.json`
- **Data**: Hospital general information and characteristics
- **Fields**: Hospital names, locations, bed counts, facility types

## 🔧 Module Details

### 1. Data Ingestion (`streaming.py`)

**Features:**
- Fetches data from multiple public APIs
- Simulates real-time streaming with configurable intervals
- Handles API rate limiting and error recovery
- Buffers data for downstream processing

**Key Classes:**
- `HospitalDataStreamer`: Main streaming class
- Methods: `fetch_hhs_data()`, `fetch_cms_data()`, `stream_data()`

### 2. Feature Engineering (`features.py`)

**Features:**
- Data cleaning and preprocessing
- Missing value handling with forward-fill
- Time-series feature creation (lags, rolling averages)
- Seasonal and cyclical encoding
- Bed utilization rate calculations

**Key Classes:**
- `HospitalFeatureEngineer`: Main feature engineering class
- Methods: `clean_data()`, `create_lag_features()`, `create_rolling_features()`

### 3. Forecasting (`forecast.py`)

**Models Implemented:**
- **ARIMA**: Classical time-series model
- **Exponential Smoothing**: Holt-Winters with seasonality
- **XGBoost**: Gradient boosting with engineered features

**Evaluation Metrics:**
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient of Determination)

**Key Classes:**
- `HospitalForecaster`: Main forecasting class
- Methods: `train_all_models()`, `make_forecasts()`, `evaluate_all_models()`

### 4. Dashboard (`dashboard.py`)

**Features:**
- Interactive Streamlit interface
- Real-time data updates
- Multiple visualization types
- Model comparison charts
- Regional heatmaps
- Auto-refresh capabilities

**Visualizations:**
- Time series plots with forecasts
- Model comparison charts
- Regional heatmaps
- Summary metrics cards
- Performance evaluation tables

## 📈 Usage Examples

### Basic Data Streaming

```python
from streaming import HospitalDataStreamer

streamer = HospitalDataStreamer()
data = streamer.fetch_hhs_data(limit=100)
print(f"Fetched {len(data)} records")
```

### Feature Engineering

```python
from features import HospitalFeatureEngineer

engineer = HospitalFeatureEngineer()
engineered_data = engineer.engineer_features(raw_data)
print(f"Engineered features: {engineer.get_feature_columns(engineered_data)}")
```

### Forecasting

```python
from forecast import HospitalForecaster

forecaster = HospitalForecaster()
models = forecaster.train_all_models(engineered_data, 'total_patients')
forecasts = forecaster.make_forecasts(engineered_data, 'total_patients', steps=7)
```

## 🎯 Target Metrics

The system forecasts four key metrics:

1. **`total_patients`**: Total hospitalized patients (adult + pediatric)
2. **`icu_patients`**: ICU patient count
3. **`bed_utilization_rate`**: Overall bed utilization percentage
4. **`icu_utilization_rate`**: ICU bed utilization percentage

## 🔍 Model Performance

### Evaluation Approach
- Time-series aware train/test splits
- Rolling window evaluation
- Multiple error metrics for comprehensive assessment

### Expected Performance
- **ARIMA**: Good for trend and seasonality
- **Exponential Smoothing**: Excellent for seasonal patterns
- **XGBoost**: Best for complex feature interactions

## 🛠️ Configuration

### Dashboard Settings
- **Data Source**: Live stream vs. sample data
- **Hospital Filter**: All hospitals or specific selection
- **Forecast Horizon**: 1-30 days
- **Model Selection**: Choose which models to display
- **Auto-refresh**: 10-60 second intervals

### API Configuration
- **Rate Limiting**: Built-in delays between requests
- **Error Handling**: Automatic retry with exponential backoff
- **Data Buffering**: Configurable buffer sizes

## 🚨 Limitations & Considerations

### Data Quality
- API data may have missing values
- Hospital reporting frequency varies
- Some hospitals may not report all metrics

### Model Limitations
- ARIMA requires sufficient historical data
- XGBoost needs feature engineering
- Seasonal patterns may change over time

### API Constraints
- Rate limits on public APIs
- Data availability depends on reporting schedules
- API endpoints may change

## 🔮 Future Enhancements

### Planned Improvements
1. **Deep Learning Models**: LSTM/GRU implementations
2. **Ensemble Methods**: Combine multiple model predictions
3. **Advanced Features**: Weather data, holiday effects
4. **Real-time Alerts**: Threshold-based notifications
5. **Database Integration**: Persistent storage solutions

### Scalability
- **Distributed Processing**: Apache Spark integration
- **Cloud Deployment**: AWS/Azure deployment options
- **Microservices**: API-first architecture

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Support

For questions or issues:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information

---

**Built with ❤️ for healthcare analytics** 