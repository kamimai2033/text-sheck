import argparse
import json
import sys

from .detect import detect


class WhitespaceTokenizer:
    def encode(self, text):
        return [tok for tok in text.split() if tok]

    def decode(self, tokens):
        return " ".join(tokens)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="SynthID Text / テキストウォーターマーク検知ツール")
    parser.add_argument("text", nargs="?", help="検査対象テキスト")
    parser.add_argument("--g-values", help="SynthID g値列 (カンマ区切り, 例: 0,1,2,3)")
    parser.add_argument("--seed", type=int, default=15485863)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--context-width", type=int, default=1)
    parser.add_argument("--z-threshold", type=float, default=4.0)
    parser.add_argument("--json", action="store_true", help="JSON で出力")
    args = parser.parse_args(argv)

    if not args.text and not args.g_values:
        parser.error("テキストまたは --g-values を指定してください")

    g_values = None
    if args.g_values:
        g_values = [int(v.strip()) for v in args.g_values.split(",") if v.strip() != ""]

    report = detect(
        args.text or "",
        tokenizer=WhitespaceTokenizer() if args.text else None,
        g_values=g_values,
        seed=args.seed,
        gamma=args.gamma,
        context_width=args.context_width,
        z_threshold=args.z_threshold,
    )

    if args.json:
        payload = {
            "text": args.text,
            "soft_watermark": None,
            "synthid": None,
            "flagged": report.is_watermarked,
        }
        if report.soft_watermark is not None:
            r = report.soft_watermark
            payload["soft_watermark"] = {
                "z_score": r.z_score, "p_value": r.p_value,
                "green_fraction": r.green_fraction,
                "num_tokens": r.num_tokens,
                "is_watermarked": r.is_watermarked,
            }
        if report.synthid is not None:
            r = report.synthid
            payload["synthid"] = {
                "chi2": r.chi2_statistic, "p_value": r.p_value,
                "num_tokens": r.num_tokens,
                "g_value_distribution": r.g_value_distribution,
                "is_watermarked": r.is_watermarked,
            }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
