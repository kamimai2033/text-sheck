import math

G_VALUE_COUNT = 4  # SynthID Text: 各トークンに 2bit の g値 (0..3)


def _lower_regularized_gamma(a, x):
    if x <= 0:
        return 0.0
    if x < a + 1.0:
        term = 1.0 / a
        total = term
        ap = a
        for _ in range(1000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        p = total * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return p
    b = x + 1.0 - a
    c = 1e300
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return 1.0 - q


def chi2_sf(x, k):
    if x <= 0:
        return 1.0
    p = _lower_regularized_gamma(k / 2.0, x / 2.0)
    return max(0.0, min(1.0, 1.0 - p))


class SynthIDResult:
    def __init__(self, chi2_statistic, p_value, num_tokens, g_value_distribution,
                 threshold=0.05):
        self.chi2_statistic = chi2_statistic
        self.p_value = p_value
        self.num_tokens = num_tokens
        self.g_value_distribution = g_value_distribution
        self.threshold = threshold
        self.is_watermarked = p_value < threshold

    def __repr__(self):
        return (
            "SynthIDResult(chi2={:.3f}, p_value={:.3e}, num_tokens={}, "
            "watermarked={})".format(
                self.chi2_statistic, self.p_value,
                self.num_tokens, self.is_watermarked,
            )
        )


class SynthIDDetector:
    def __init__(self, g_value_count=G_VALUE_COUNT, threshold=0.05):
        self.g_value_count = g_value_count
        self.threshold = threshold

    def detect(self, g_values):
        counts = [0] * self.g_value_count
        for v in g_values:
            if not (0 <= v < self.g_value_count):
                raise ValueError("不正な g値: {}".format(v))
            counts[v] += 1
        n = len(g_values)
        if n == 0:
            raise ValueError("g値列が空です")
        expected = n / float(self.g_value_count)
        chi2 = sum((c - expected) ** 2 / expected for c in counts)
        dof = self.g_value_count - 1
        p_value = chi2_sf(chi2, dof)
        distribution = [c / float(n) for c in counts]
        return SynthIDResult(chi2, p_value, n, distribution, self.threshold)
