"""
tests/test_predict.py
----------------------
Unit tests for the HousePricePredictor pipeline.

Run with:
    python -m pytest tests/ -v
"""

import pytest
from src.predict import HousePricePredictor


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def predictor():
    """Create one predictor instance shared across all tests."""
    return HousePricePredictor()


@pytest.fixture
def sf_house():
    """A typical San Francisco house — should predict high price."""
    return {
        'MedInc'    : 8.5,
        'HouseAge'  : 15.0,
        'AveRooms'  : 7.0,
        'AveBedrms' : 1.2,
        'Population': 800.0,
        'AveOccup'  : 2.5,
        'Latitude'  : 37.77,
        'Longitude' : -122.41
    }


@pytest.fixture
def inland_house():
    """A typical inland low-income house — should predict lower price."""
    return {
        'MedInc'    : 2.0,
        'HouseAge'  : 40.0,
        'AveRooms'  : 4.0,
        'AveBedrms' : 1.0,
        'Population': 1200.0,
        'AveOccup'  : 3.5,
        'Latitude'  : 36.50,
        'Longitude' : -119.50
    }


# ── Tests: Output Structure ───────────────────────────────────────────

class TestPredictorOutput:

    def test_returns_dict(self, predictor, sf_house):
        """Prediction must return a dictionary."""
        result = predictor.predict(sf_house)
        assert isinstance(result, dict)

    def test_all_keys_present(self, predictor, sf_house):
        """Result must contain all expected keys."""
        result = predictor.predict(sf_house)
        expected_keys = {
            'predicted_price',
            'predicted_price_100k',
            'confidence_range',
            'model_name'
        }
        assert expected_keys.issubset(result.keys())

    def test_price_is_positive(self, predictor, sf_house):
        """Predicted price must be a positive number."""
        result = predictor.predict(sf_house)
        assert result['predicted_price'] > 0

    def test_confidence_range_valid(self, predictor, sf_house):
        """Low bound must be less than high bound."""
        result = predictor.predict(sf_house)
        low, high = result['confidence_range']
        assert low < result['predicted_price'] < high

    def test_price_in_realistic_range(self, predictor, sf_house):
        """Price must be within a realistic California range ($50k–$5M)."""
        result = predictor.predict(sf_house)
        assert 50_000 < result['predicted_price'] < 5_000_000


# ── Tests: Business Logic ─────────────────────────────────────────────

class TestPredictorLogic:

    def test_sf_more_expensive_than_inland(
        self, predictor, sf_house, inland_house
    ):
        """
        SF house should predict higher than inland house.
        This validates the model learned geography correctly.
        """
        sf_price     = predictor.predict(sf_house)['predicted_price']
        inland_price = predictor.predict(inland_house)['predicted_price']
        assert sf_price > inland_price, (
            f"Expected SF (${sf_price:,.0f}) > Inland (${inland_price:,.0f})"
        )

    def test_higher_income_increases_price(self, predictor, sf_house):
        """
        Increasing median income should increase predicted price.
        This validates the model learned income-price relationship.
        """
        low_income_house  = {**sf_house, 'MedInc': 1.0}
        high_income_house = {**sf_house, 'MedInc': 12.0}

        low_price  = predictor.predict(low_income_house)['predicted_price']
        high_price = predictor.predict(high_income_house)['predicted_price']

        assert high_price > low_price, (
            f"Higher income should predict higher price. "
            f"Got low={low_price:,.0f}, high={high_price:,.0f}"
        )

    def test_batch_predict(self, predictor, sf_house, inland_house):
        """Batch prediction must return list of correct length."""
        results = predictor.predict_batch([sf_house, inland_house])
        assert isinstance(results, list)
        assert len(results) == 2
        assert all('predicted_price' in r for r in results)


# ── Tests: Input Validation ───────────────────────────────────────────

class TestInputValidation:

    def test_missing_field_raises_error(self, predictor):
        """Missing required field must raise ValueError."""
        incomplete = {
            'MedInc'    : 5.0,
            'HouseAge'  : 20.0,
            # Missing all other fields
        }
        with pytest.raises(ValueError, match="Missing required fields"):
            predictor.predict(incomplete)

    def test_negative_value_raises_error(self, predictor, sf_house):
        """Negative feature value must raise ValueError."""
        bad_input = {**sf_house, 'MedInc': -1.0}
        with pytest.raises(ValueError):
            predictor.predict(bad_input)

    def test_invalid_latitude_raises_error(self, predictor, sf_house):
        """Latitude outside California must raise ValueError."""
        bad_input = {**sf_house, 'Latitude': 10.0}
        with pytest.raises(ValueError, match="Latitude"):
            predictor.predict(bad_input)

    def test_invalid_longitude_raises_error(self, predictor, sf_house):
        """Longitude outside California must raise ValueError."""
        bad_input = {**sf_house, 'Longitude': -70.0}
        with pytest.raises(ValueError, match="Longitude"):
            predictor.predict(bad_input)

    def test_bedrooms_exceed_rooms_raises_error(self, predictor, sf_house):
        """Bedrooms > Rooms is physically impossible — must raise error."""
        bad_input = {**sf_house, 'AveBedrms': 10.0, 'AveRooms': 3.0}
        with pytest.raises(ValueError, match="AveBedrms"):
            predictor.predict(bad_input)