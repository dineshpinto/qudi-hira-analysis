import logging
from unittest import TestCase

from qudi_hira_analysis.helper_functions import (
    decibelm_to_watts,
    format_exponent_as_str,
    log_tick_formatter,
)

logging.disable(logging.CRITICAL)


class TestFitting(TestCase):
    def test_decibelm_to_watts(self):
        self.assertEqual(0.01, decibelm_to_watts(10))

    def test_format_exponent_as_str(self):
        self.assertEqual(r"${ 1.0 } \cdot 10^{ -4 }$", format_exponent_as_str(0.0001))

    def test_log_tick_formatter(self):
        self.assertEqual(r"$10^{1}$", log_tick_formatter(1))
