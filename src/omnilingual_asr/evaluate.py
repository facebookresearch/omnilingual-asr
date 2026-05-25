# omnilingual_asr/evaluate.py

from jiwer import wer, cer
from typing import Optional, List, Tuple
import pandas as pd
from collections import defaultdict

def safe_wer(reference: str, hypothesis: str) -> Optional[float]:
    """Compute WER; returns None if inputs are empty."""
    if not reference or not hypothesis:
        return None
    try:
        return wer(reference, hypothesis)
    except Exception:
        return None

def safe_cer(reference: str, hypothesis: str) -> Optional[float]:
    """Compute CER; returns None if inputs are empty."""
    if not reference or not hypothesis:
        return None
    try:
        return cer(reference, hypothesis)
    except Exception:
        return None

def add_metrics_columns(df: pd.DataFrame, ref_col: str, hyp_col: str) -> pd.DataFrame:
    """Add 'wer' and 'cer' columns (floats in [0,1]) to the DataFrame."""
    df = df.copy()
    df["wer"] = df.apply(lambda row: safe_wer(row[ref_col], row[hyp_col]), axis=1)
    df["cer"] = df.apply(lambda row: safe_cer(row[ref_col], row[hyp_col]), axis=1)
    return df

def idiom_summary(df: pd.DataFrame, idiom_col: str = "idiom") -> pd.DataFrame:
    """
    Build a per‑idiom summary DataFrame with columns:
    idiom, samples, wer_mean, wer_std, cer_mean, cer_std.
    Percentages are multiplied by 100.
    """
    agg = df.groupby(idiom_col).agg(
        samples=(idiom_col, "count"),
        wer_mean=("wer", "mean"),
        wer_std=("wer", "std"),
        cer_mean=("cer", "mean"),
        cer_std=("cer", "std"),
    ).reset_index()

    # Add an OVERALL row
    overall = pd.DataFrame([{
        idiom_col: "OVERALL",
        "samples": len(df),
        "wer_mean": df["wer"].mean(),
        "wer_std": df["wer"].std(),
        "cer_mean": df["cer"].mean(),
        "cer_std": df["cer"].std(),
    }])

    summary = pd.concat([agg, overall], ignore_index=True)

    # Convert to percentages and round
    for col in ["wer_mean", "wer_std", "cer_mean", "cer_std"]:
        summary[col] = (summary[col] * 100).round(2)

    return summary

def print_evaluation_summary(summary: pd.DataFrame) -> None:
    """Print a formatted summary table and per‑idiom results."""
    print("\n" + "=" * 50)
    print("OVERALL RESULTS")
    print("=" * 50)
    overall = summary[summary["idiom"] == "OVERALL"].iloc[0]
    print(f"Total test samples: {overall['samples']}")
    print(f"Word Error Rate (WER): {overall['wer_mean']:.2f}%")
    print(f"Character Error Rate (CER): {overall['cer_mean']:.2f}%")

    print("\n" + "=" * 50)
    print("PER IDIOM RESULTS")
    print("=" * 50)
    for _, row in summary[summary["idiom"] != "OVERALL"].iterrows():
        print(f"\n{row['idiom'].upper()}")
        print(f"  Samples: {row['samples']}")
        print(f"  WER: {row['wer_mean']:.2f}%")
        print(f"  CER: {row['cer_mean']:.2f}%")

    print("\n" + "=" * 50)
    print("SUMMARY TABLE")
    print("=" * 50)
    print(summary.to_string(index=False))

def show_examples(df: pd.DataFrame, hyp_col: str, ref_col: str = "sentence", n: int = 5) -> None:
    """Print a few random examples with WER/CER."""
    import random
    random.seed(42)
    sample_indices = random.sample(range(len(df)), min(n, len(df)))

    print("\n" + "=" * 60)
    print("EXAMPLE TRANSCRIPTIONS")
    print("=" * 60)
    for idx in sample_indices:
        row = df.iloc[idx]
        print(f"\nIdiom: {row['idiom']}")
        print(f"Reference:    {row[ref_col][:150]}")
        print(f"Hypothesis:   {row[hyp_col][:150]}")
        print(f"WER: {row['wer']*100:.1f}% | CER: {row['cer']*100:.1f}%")
        print("-" * 40)