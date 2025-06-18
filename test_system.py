"""
Test script for Hospital Bed Occupancy Forecasting System

This script tests all modules to ensure they work together correctly.
"""

import sys
import traceback
from datetime import datetime

def test_streaming():
    """Test the streaming module."""
    print("🔍 Testing streaming module...")
    try:
        from streaming import HospitalDataStreamer
        
        streamer = HospitalDataStreamer()
        data = streamer.fetch_hhs_data(limit=10)
        
        if not data.empty:
            print(f"✅ Streaming test passed: Fetched {len(data)} records")
            print(f"   Columns: {list(data.columns)}")
            return True
        else:
            print("❌ Streaming test failed: No data received")
            return False
            
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        traceback.print_exc()
        return False

def test_features():
    """Test the feature engineering module."""
    print("🔍 Testing feature engineering module...")
    try:
        from features import HospitalFeatureEngineer
        from streaming import HospitalDataStreamer
        
        # Get sample data
        streamer = HospitalDataStreamer()
        raw_data = streamer.fetch_hhs_data(limit=50)
        
        if raw_data.empty:
            print("❌ Feature engineering test failed: No raw data available")
            return False
        
        # Engineer features
        engineer = HospitalFeatureEngineer()
        engineered_data = engineer.engineer_features(raw_data)
        
        if not engineered_data.empty:
            feature_columns = engineer.get_feature_columns(engineered_data)
            print(f"✅ Feature engineering test passed: {len(engineered_data)} records, {len(engineered_data.columns)} columns")
            print(f"   Feature columns: {len(feature_columns)}")
            return True
        else:
            print("❌ Feature engineering test failed: No engineered data produced")
            return False
            
    except Exception as e:
        print(f"❌ Feature engineering test failed: {e}")
        traceback.print_exc()
        return False

def test_forecasting():
    """Test the forecasting module."""
    print("🔍 Testing forecasting module...")
    try:
        from forecast import HospitalForecaster
        from features import HospitalFeatureEngineer
        from streaming import HospitalDataStreamer
        
        # Get and engineer data
        streamer = HospitalDataStreamer()
        raw_data = streamer.fetch_hhs_data(limit=100)
        
        if raw_data.empty:
            print("❌ Forecasting test failed: No raw data available")
            return False
        
        engineer = HospitalFeatureEngineer()
        engineered_data = engineer.engineer_features(raw_data)
        
        if engineered_data.empty:
            print("❌ Forecasting test failed: No engineered data available")
            return False
        
        # Test forecasting
        forecaster = HospitalForecaster()
        models = forecaster.train_all_models(engineered_data, 'total_patients')
        
        if models:
            forecasts = forecaster.make_forecasts(engineered_data, 'total_patients', steps=3)
            print(f"✅ Forecasting test passed: {len(models)} models trained, {len(forecasts)} forecasts generated")
            return True
        else:
            print("❌ Forecasting test failed: No models trained")
            return False
            
    except Exception as e:
        print(f"❌ Forecasting test failed: {e}")
        traceback.print_exc()
        return False

def test_dashboard_import():
    """Test that dashboard can be imported."""
    print("🔍 Testing dashboard module import...")
    try:
        from dashboard import HospitalDashboard
        
        dashboard = HospitalDashboard()
        print("✅ Dashboard import test passed")
        return True
        
    except Exception as e:
        print(f"❌ Dashboard import test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🏥 Hospital Bed Occupancy Forecasting System - Test Suite")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    print()
    
    tests = [
        ("Streaming Module", test_streaming),
        ("Feature Engineering Module", test_features),
        ("Forecasting Module", test_forecasting),
        ("Dashboard Module", test_dashboard_import)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nTo run the dashboard:")
        print("  streamlit run dashboard.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 