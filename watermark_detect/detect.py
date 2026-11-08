from .soft_watermark import SoftWatermarkDetector
from .synthid import SynthIDDetector


class DetectionReport:
    def __init__(self, text, soft_result, synthid_result, synthid_g_values=None):
        self.text = text
        self.soft_watermark = soft_result
        self.synthid = synthid_result
        self.synthid_g_values = synthid_g_values

    @property
    def is_watermarked(self):
        flagged = []
        if self.soft_watermark is not None and self.soft_watermark.is_watermarked:
            flagged.append("soft-watermark")
        if self.synthid is not None and self.synthid.is_watermarked:
            flagged.append("synthid-text")
        return flagged

    def summary(self):
        lines = ["=== 検知結果 ==="]
        lines.append("テキスト: {}...".format(self.text[:60]))
        if self.soft_watermark is not None:
            r = self.soft_watermark
            lines.append(
                "[Soft Watermark] z={:.3f} p={:.3e} green率={:.3f} 判定={}".format(
                    r.z_score, r.p_value, r.green_fraction,
                    "検出" if r.is_watermarked else "未検出"))
        if self.synthid is not None:
            r = self.synthid
            lines.append(
                "[SynthID Text] chi2={:.3f} p={:.3e} 判定={}".format(
                    r.chi2_statistic, r.p_value,
                    "検出" if r.is_watermarked else "未検出"))
        flagged = self.is_watermarked
        lines.append("総合判定: {}".format(", ".join(flagged) if flagged else "ウォーターマークなし"))
        return "\n".join(lines)

    def __str__(self):
        return self.summary()


def detect(text, tokenizer=None, g_values=None, seed=15485863, gamma=0.5,
           context_width=1, z_threshold=4.0, chi2_threshold=0.05):
    soft_result = None
    if tokenizer is not None:
        detector = SoftWatermarkDetector(
            tokenizer, seed=seed, gamma=gamma,
            context_width=context_width, threshold=z_threshold)
        soft_result = detector.detect(text)
    synthid_result = None
    if g_values is not None:
        synthid_result = SynthIDDetector(threshold=chi2_threshold).detect(g_values)
    if soft_result is None and synthid_result is None:
        raise ValueError("tokenizer または g_values のいずれかを指定してください")
    return DetectionReport(text, soft_result, synthid_result, g_values)
