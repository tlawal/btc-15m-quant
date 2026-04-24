"""
Tier 2 #9: unit tests for time-bucketed isotonic calibration.

Covers:
- bucket dispatch (minutes → bucket name)
- backward-compatible 1-arg calibrate() call
- 2-arg calibrate(p, min_rem) routes through the right bucket model
- missing bucket falls back to global model
- all missing ⇒ identity
"""

import pytest
import calibration as cal


class TestBucketDispatch:
    def test_zero_to_two(self):
        assert cal._bucket_for(0.0)  == "bucket_0_2"
        assert cal._bucket_for(1.9)  == "bucket_0_2"

    def test_two_to_five(self):
        assert cal._bucket_for(2.0) == "bucket_2_5"
        assert cal._bucket_for(4.99) == "bucket_2_5"

    def test_five_to_ten(self):
        assert cal._bucket_for(5.0) == "bucket_5_10"
        assert cal._bucket_for(9.99) == "bucket_5_10"

    def test_ten_plus(self):
        assert cal._bucket_for(10.0) == "bucket_10plus"
        assert cal._bucket_for(14.99) == "bucket_10plus"

    def test_none(self):
        assert cal._bucket_for(None) is None

    def test_nan_string(self):
        assert cal._bucket_for("nope") is None


class TestCalibrateFallback:
    def setup_method(self, _):
        # Reset module-level state
        cal._global_model = None
        cal._bucket_models = {}

    def test_identity_when_no_models(self):
        """With nothing loaded, calibrate is identity."""
        assert cal.calibrate(0.73) == 0.73
        assert cal.calibrate(0.73, min_rem=1.5) == 0.73

    def test_backcompat_single_arg(self):
        """Legacy 1-arg call must still work."""
        # A stub "model" that always predicts 0.5
        class StubModel:
            def predict(self, X): return [0.5]
        cal._global_model = StubModel()
        assert abs(cal.calibrate(0.9) - 0.5) < 1e-6

    def test_bucket_routing(self):
        """calibrate(p, min_rem) picks the correct bucket model."""
        class GlobalStub:
            def predict(self, X): return [0.3]
        class Bucket02Stub:
            def predict(self, X): return [0.8]
        cal._global_model = GlobalStub()
        cal._bucket_models = {"bucket_0_2": Bucket02Stub()}
        # Routed to bucket
        assert abs(cal.calibrate(0.5, min_rem=1.0) - 0.8) < 1e-6
        # No matching bucket → falls back to global
        assert abs(cal.calibrate(0.5, min_rem=7.0) - 0.3) < 1e-6
        # min_rem=None → global
        assert abs(cal.calibrate(0.5) - 0.3) < 1e-6


class TestRealModels:
    """Smoke test using the models committed during training (if present)."""

    def test_real_models_load_and_calibrate(self):
        ok = cal.load_calibration_model()
        if not ok:
            pytest.skip("No calibration models on disk (run `python calibration.py --source logs`)")
        # Sanity: with models loaded, we should get different outputs across buckets
        raw = 0.85
        out_sprint = cal.calibrate(raw, min_rem=1.0)
        out_mid    = cal.calibrate(raw, min_rem=7.0)
        out_early  = cal.calibrate(raw, min_rem=12.0)
        # All in valid probability range
        for o in (out_sprint, out_mid, out_early):
            assert 0.0 < o < 1.0
        # At least one bucket differs from the raw probability
        assert any(abs(o - raw) > 1e-4 for o in (out_sprint, out_mid, out_early))
