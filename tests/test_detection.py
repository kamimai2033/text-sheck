import math
import random
import unittest

from watermark_detect.detect import detect
from watermark_detect.hashing import is_green
from watermark_detect.soft_watermark import SoftWatermarkDetector
from watermark_detect.synthid import SynthIDDetector, chi2_sf


class IntTokenizer:
    def encode(self, text):
        return [int(t) for t in text.split()]


class TestHashing(unittest.TestCase):
    def test_green_fraction_close_to_gamma(self):
        rng = random.Random(42)
        n = 20000
        hits = sum(is_green(rng.randrange(100000), [rng.randrange(100000)])
                   for _ in range(n))
        self.assertAlmostEqual(hits / n, 0.5, delta=0.02)

    def test_deterministic(self):
        self.assertEqual(is_green(123, [456]), is_green(123, [456]))


class TestSoftWatermark(unittest.TestCase):
    def test_watermarked_text_detected(self):
        rng = random.Random(7)
        tokens = [rng.randrange(100000)]
        for _ in range(300):
            candidates = [rng.randrange(100000) for _ in range(20)]
            green = [c for c in candidates if is_green(c, tokens[-1:])]
            tokens.append(green[0] if green else candidates[0])
        text = " ".join(str(t) for t in tokens)
        result = SoftWatermarkDetector(IntTokenizer()).detect(text)
        self.assertGreater(result.z_score, 4.0)
        self.assertTrue(result.is_watermarked)
        self.assertLess(result.p_value, 1e-4)

    def test_random_text_not_flagged(self):
        rng = random.Random(1)
        tokens = " ".join(str(rng.randrange(100000)) for _ in range(300))
        result = SoftWatermarkDetector(IntTokenizer()).detect(tokens)
        self.assertFalse(result.is_watermarked)

    def test_short_text_raises(self):
        with self.assertRaises(ValueError):
            SoftWatermarkDetector(IntTokenizer()).detect("1")


class TestSynthID(unittest.TestCase):
    def test_uniform_g_values_not_flagged(self):
        rng = random.Random(3)
        g = [rng.randrange(4) for _ in range(4000)]
        result = SynthIDDetector().detect(g)
        self.assertGreater(result.p_value, 0.05)
        self.assertFalse(result.is_watermarked)

    def test_skewed_g_values_flagged(self):
        g = [0] * 3000 + [1] * 300 + [2] * 300 + [3] * 400
        result = SynthIDDetector().detect(g)
        self.assertLess(result.p_value, 1e-10)
        self.assertTrue(result.is_watermarked)

    def test_invalid_g_value(self):
        with self.assertRaises(ValueError):
            SynthIDDetector().detect([0, 4])

    def test_chi2_sf_known_values(self):
        self.assertAlmostEqual(chi2_sf(3.84, 1), 0.05, places=2)
        self.assertAlmostEqual(chi2_sf(0.0, 3), 1.0)


class TestCombined(unittest.TestCase):
    def test_detect_json_flags(self):
        rng = random.Random(7)
        tokens = [rng.randrange(100000)]
        for _ in range(300):
            candidates = [rng.randrange(100000) for _ in range(20)]
            green = [c for c in candidates if is_green(c, tokens[-1:])]
            tokens.append(green[0] if green else candidates[0])
        text = " ".join(str(t) for t in tokens)
        report = detect(text, tokenizer=IntTokenizer())
        self.assertIn("soft-watermark", report.is_watermarked)
        self.assertIn("検知結果", report.summary())


if __name__ == "__main__":
    unittest.main()
