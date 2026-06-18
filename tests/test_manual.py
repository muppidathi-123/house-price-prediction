from src.predict import HousePricePredictor

predictor = HousePricePredictor()

# Test house 1
result = predictor.predict({
    'MedInc': 8.5,
    'HouseAge': 15.0,
    'AveRooms': 7.0,
    'AveBedrms': 1.2,
    'Population': 800.0,
    'AveOccup': 2.5,
    'Latitude': 37.77,
    'Longitude': -122.41
})

print("Test 1 — San Francisco house:")
print(f"Predicted: ${result['predicted_price']:,.0f}")
print(f"Range: ${result['confidence_range'][0]:,.0f} - ${result['confidence_range'][1]:,.0f}")
print(f"Model: {result['model_name']}")

print()

# Test house 2
result2 = predictor.predict({
    'MedInc': 2.0,
    'HouseAge': 40.0,
    'AveRooms': 4.0,
    'AveBedrms': 1.0,
    'Population': 1200.0,
    'AveOccup': 3.5,
    'Latitude': 36.50,
    'Longitude': -119.50
})

print("Test 2 — Inland low-income area:")
print(f"Predicted: ${result2['predicted_price']:,.0f}")
print(f"Range: ${result2['confidence_range'][0]:,.0f} - ${result2['confidence_range'][1]:,.0f}")
print(f"Model: {result2['model_name']}")