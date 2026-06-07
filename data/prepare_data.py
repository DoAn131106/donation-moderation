"""
prepare_data.py
───────────────
Builds the full training dataset from 3 sources:

  1. ViHSD  (HuggingFace, ~33k rows) — covers insult + profanity
  2. ViCTSD (local CSV, ~10k rows)   — covers additional insult cases
  3. Synthetic .txt files            — covers spam + political_review

Label mapping:
  ViHSD  CLEAN (0)      → all = 0
  ViHSD  OFFENSIVE (1)  → insult=1, profanity=1
  ViHSD  HATE (2)       → insult=1
  ViCTSD TOXIC (1)      → insult=1
  ViCTSD NONE (0)       → all = 0
  synthetic_spam        → spam=1
  synthetic_political   → political_review=1

Synthetic data is split 70/15/15 into train/valid/test so that
spam and political_review labels are represented in all three splits.

Expected folder structure before running:
  data/
  ├── ViCTSD_train.csv          ← download from github.com/tarudesu/ViCTSD
  ├── ViCTSD_valid.csv          ← download from github.com/tarudesu/ViCTSD
  ├── ViCTSD_test.csv           ← download from github.com/tarudesu/ViCTSD
  ├── synthetic_spam.txt        ← one message per line, spam examples
  └── synthetic_political.txt   ← one message per line, political examples

Usage:
  pip install datasets pandas scikit-learn
  python prepare_data.py
"""

import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42

# Synthetic split ratios — must sum to 1.0
SYN_TRAIN = 0.70
SYN_VALID = 0.15
SYN_TEST  = 0.15


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD ViHSD
# ══════════════════════════════════════════════════════════════════════════════

VIHSD_ID    = "ura-hcmut/UIT-ViHSD"   # public mirror, no login required
VIHSD_TEXT  = "free_text"
VIHSD_LABEL = "label_id"              # 0=CLEAN, 1=OFFENSIVE, 2=HATE

print(f"[1/3] Downloading ViHSD ({VIHSD_ID})...")

vihsd_train = load_dataset(VIHSD_ID, split="train")
vihsd_valid = load_dataset(VIHSD_ID, split="validation")
vihsd_test  = load_dataset(VIHSD_ID, split="test")

def map_vihsd_row(row):
    label = row[VIHSD_LABEL]
    return {
        "text":             row[VIHSD_TEXT],
        "insult":           1 if label in (1, 2) else 0,
        "profanity":        1 if label == 1 else 0,
        "political_review": 0,
        "spam":             0,
    }

def convert_vihsd(dataset):
    return pd.DataFrame([map_vihsd_row(r) for r in dataset])

df_vihsd_train = convert_vihsd(vihsd_train)
df_vihsd_valid = convert_vihsd(vihsd_valid)
df_vihsd_test  = convert_vihsd(vihsd_test)

print(f"   ViHSD → train: {len(df_vihsd_train):,}  valid: {len(df_vihsd_valid):,}  test: {len(df_vihsd_test):,}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD ViCTSD
# ══════════════════════════════════════════════════════════════════════════════
#
# Download from: https://github.com/tarudesu/ViCTSD
# Place in data/: ViCTSD_train.csv, ViCTSD_valid.csv, ViCTSD_test.csv
#
# Columns: Sentence (text), Toxicity (0=NONE, 1=TOXIC)

print("\n[2/3] Loading ViCTSD from local CSV files...")

VICTSD_TEXT  = "Sentence"
VICTSD_LABEL = "Toxicity"

def load_victsd(path):
    if not os.path.exists(path):
        print(f"   [SKIP] {path} not found.")
        return pd.DataFrame()

    df = pd.read_csv(path)

    for col in [VICTSD_TEXT, VICTSD_LABEL]:
        if col not in df.columns:
            raise ValueError(
                f"Expected column '{col}' not found in {path}.\n"
                f"Found: {df.columns.tolist()}\n"
                "Make sure you downloaded the correct files from github.com/tarudesu/ViCTSD"
            )

    return pd.DataFrame({
        "text":             df[VICTSD_TEXT].fillna("").astype(str),
        "insult":           df[VICTSD_LABEL].astype(int),
        "profanity":        0,
        "political_review": 0,
        "spam":             0,
    })

df_victsd_train = load_victsd("data/ViCTSD_train.csv")
df_victsd_valid = load_victsd("data/ViCTSD_valid.csv")
df_victsd_test  = load_victsd("data/ViCTSD_test.csv")

if not df_victsd_train.empty:
    print(f"   ViCTSD → train: {len(df_victsd_train):,}  valid: {len(df_victsd_valid):,}  test: {len(df_victsd_test):,}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. LOAD + SPLIT SYNTHETIC DATA
# ══════════════════════════════════════════════════════════════════════════════
#
# Synthetic data is split 70/15/15 

print("\n[3/3] Loading and splitting synthetic data...")

def load_and_split_synthetic(path, label_col):
    """
    Reads a .txt file, builds a DataFrame, then splits 70/15/15
    into train/valid/test with stratification on label_col.
    Returns (df_train, df_valid, df_test).
    """
    if not os.path.exists(path):
        print(f"   [SKIP] {path} not found.")
        empty = pd.DataFrame()
        return empty, empty, empty

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    df = pd.DataFrame({
        "text":             lines,
        "insult":           0,
        "profanity":        0,
        "political_review": 0,
        "spam":             0,
    })
    df[label_col] = 1

    # First cut: train vs temp (valid+test)
    df_tr, df_temp = train_test_split(
        df,
        test_size=(SYN_VALID + SYN_TEST),
        random_state=RANDOM_SEED,
    )

    # Second cut: split temp into valid and test equally
    valid_ratio_of_temp = SYN_VALID / (SYN_VALID + SYN_TEST)
    df_va, df_te = train_test_split(
        df_temp,
        test_size=(1 - valid_ratio_of_temp),
        random_state=RANDOM_SEED,
    )

    print(f"   {path} ({label_col}=1) → train: {len(df_tr)}  valid: {len(df_va)}  test: {len(df_te)}")
    return df_tr, df_va, df_te

df_syn_spam_train, df_syn_spam_valid, df_syn_spam_test           = load_and_split_synthetic(
    "data/synthetic_spam.txt", "spam"
)
df_syn_pol_train,  df_syn_pol_valid,  df_syn_pol_test            = load_and_split_synthetic(
    "data/synthetic_political.txt", "political_review"
)


# ══════════════════════════════════════════════════════════════════════════════
# 4. MERGE ALL SOURCES
# ══════════════════════════════════════════════════════════════════════════════

def merge(*dfs):
    return pd.concat(
        [df for df in dfs if not df.empty],
        ignore_index=True,
    )

df_train = merge(df_vihsd_train, df_victsd_train, df_syn_spam_train, df_syn_pol_train)
df_valid = merge(df_vihsd_valid, df_victsd_valid, df_syn_spam_valid, df_syn_pol_valid)
df_test  = merge(df_vihsd_test,  df_victsd_test,  df_syn_spam_test,  df_syn_pol_test)


# ══════════════════════════════════════════════════════════════════════════════
# 5. SAVE
# ══════════════════════════════════════════════════════════════════════════════

os.makedirs("data", exist_ok=True)

df_train.to_csv("data/train.csv", index=False, encoding="utf-8")
df_valid.to_csv("data/valid.csv", index=False, encoding="utf-8")
df_test.to_csv("data/test.csv",   index=False, encoding="utf-8")

print(f"\nSaved:")
print(f"  data/train.csv  ({len(df_train):,} rows)")
print(f"  data/valid.csv  ({len(df_valid):,} rows)")
print(f"  data/test.csv   ({len(df_test):,} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# 6. FINAL LABEL DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
#
# What you want to see:
#   insult + profanity   → healthy % from ViHSD + ViCTSD
#   spam + political     → small % from synthetic — that's expected and fine,
#                          pos_weight in train_phobert.py will compensate

def print_distribution(df, split_name):
    print(f"\n  {split_name}:")
    for col in ["insult", "profanity", "political_review", "spam"]:
        pos   = int(df[col].sum())
        total = len(df)
        print(f"    {col:20s}: {pos:>5,} / {total:,} ({pos/total*100:.1f}%)")

print("\nLabel distribution:")
print_distribution(df_train, "train")
print_distribution(df_valid, "valid")
print_distribution(df_test,  "test ")
