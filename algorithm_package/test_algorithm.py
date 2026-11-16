import unittest
from datetime import datetime
from algorithm import (
    init_bins, tick, snapshot_bins, BIN_LOCATIONS,
    predict_hourly_increment,
)


class TestRealtimeAlgorithm(unittest.TestCase):
    def test_init_defines_five_bins_at_zero(self):
        bins = init_bins(start_time="2026-07-01 00:00:00")
        self.assertEqual(set(bins.keys()), set(BIN_LOCATIONS.keys()))
        self.assertTrue(all(b["fillRate"] == 0.0 for b in bins.values()))

    def test_predictor_calls_model_and_returns_nonnegative_increment(self):
        prediction = predict_hourly_increment("A001", "2026-07-01 01:00:00")
        self.assertGreaterEqual(prediction, 0.0)
        self.assertLessEqual(prediction, 100.0)

    def test_one_hour_tick_matches_model_increment(self):
        expected = predict_hourly_increment("A001", "2026-07-01 01:00:00")
        bins = init_bins(start_time="2026-07-01 00:00:00")
        tick(bins, elapsed_minutes=60)
        self.assertAlmostEqual(bins["A001"]["fillRate"], expected, places=4)

    def test_clean_flag_resets_only_marked_bin(self):
        bins = init_bins(initial_fill={"A001": 70, "A002": 40}, start_time="2026-07-01 12:00:00")
        tick(bins, clean_flags={"A001": True}, elapsed_minutes=5)
        self.assertEqual(bins["A001"]["fillRate"], 0.0)
        self.assertGreater(bins["A002"]["fillRate"], 40.0)

    def test_sensor_reading_overrides_model(self):
        bins = init_bins(start_time="2026-07-01 00:00:00")
        tick(bins, sensor_readings={"A001": 48.5}, elapsed_minutes=5)
        self.assertEqual(bins["A001"]["fillRate"], 48.5)

    def test_incoming_amounts_is_increment_in_percentage_points(self):
        bins = init_bins(initial_fill=10, start_time="2026-07-01 00:00:00")
        tick(bins, incoming_amounts={"A001": 2.25}, elapsed_minutes=5)
        self.assertEqual(bins["A001"]["fillRate"], 12.25)

    def test_fill_rate_and_risk_stay_in_range(self):
        bins = init_bins(start_time="2026-07-01 00:00:00")
        for _ in range(24 * 60 // 5):
            tick(bins, elapsed_minutes=5)
        for state in bins.values():
            self.assertGreaterEqual(state["fillRate"], 0)
            self.assertLessEqual(state["fillRate"], 100)
            self.assertGreaterEqual(state["risk"], 0)
            self.assertLessEqual(state["risk"], 100)
            self.assertIn(state["riskLevel"], {"low", "medium", "high", "critical"})

    def test_snapshot_hides_internal_fields(self):
        bins = init_bins(start_time="2026-07-01 00:00:00")
        rows = snapshot_bins(bins)
        self.assertEqual(len(rows), 5)
        self.assertTrue(all("_clock" not in row for row in rows))

    def test_zero_default_gain_is_not_hardcoded_375x_curve(self):
        base = predict_hourly_increment("A001", "2026-07-01 01:00:00", gain=1.0)
        zero_gain = predict_hourly_increment("A001", "2026-07-01 01:00:00", gain=0.0)
        self.assertGreater(base, 0.0)
        self.assertEqual(zero_gain, 0.0)


if __name__ == "__main__":
    unittest.main()
