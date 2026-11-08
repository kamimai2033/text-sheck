import math

from .hashing import DEFAULT_GAMMA, DEFAULT_SEED, is_green


class SoftWatermarkResult:
    def __init__(self, z_score, p_value, num_tokens, green_fraction, gamma, threshold=4.0):
        self.z_score = z_score
        self.p_value = p_value
        self.num_tokens = num_tokens
        self.green_fraction = green_fraction
        self.gamma = gamma
        self.threshold = threshold
        self.is_watermarked = z_score >= threshold

    def __repr__(self):
        return (
            "SoftWatermarkResult(z_score={:.3f}, p_value={:.3e}, "
            "green_fraction={:.3f}, num_tokens={}, watermarked={})".format(
                self.z_score, self.p_value, self.green_fraction,
                self.num_tokens, self.is_watermarked,
            )
        )


class SoftWatermarkDetector:
    def __init__(self, tokenizer, seed=DEFAULT_SEED, gamma=DEFAULT_GAMMA,
                 context_width=1, threshold=4.0):
        self.tokenizer = tokenizer
        self.seed = seed
        self.gamma = gamma
        self.context_width = context_width
        self.threshold = threshold

    def detect(self, text):
        token_ids = self.tokenizer.encode(text)
        return self.detect_tokens(token_ids)

    def detect_tokens(self, token_ids):
        n = len(token_ids)
        if n <= self.context_width:
            raise ValueError("テキストが短すぎます: {} トークン".format(n))
        t = n - self.context_width
        hits = 0
        for i in range(self.context_width, n):
            prev = token_ids[max(0, i - self.context_width):i]
            if is_green(token_ids[i], prev, self.seed, self.gamma):
                hits += 1
        green_fraction = hits / float(t)
        mean = self.gamma * t
        sd = math.sqrt(t * self.gamma * (1.0 - self.gamma))
        z_score = (hits - mean) / sd if sd > 0 else 0.0
        p_value = min(1.0, math.erfc(z_score / math.sqrt(2.0)))
        return SoftWatermarkResult(z_score, p_value, t, green_fraction,
                                   self.gamma, self.threshold)
