import pandas as pd
import numpy as np

# ==========================
# Load Dataset
# ==========================
FILE = "data/processed/tabular_features.csv"   # Change path if needed

df = pd.read_csv(FILE)

print("=" * 80)
print("UEBA FEATURE DATASET VALIDATION")
print("=" * 80)

# =====================================================
# 1. Basic Statistics
# =====================================================
print("\n[1] Dataset Statistics")
print(f"Rows    : {len(df)}")
print(f"Columns : {len(df.columns)}")

# =====================================================
# 2. Missing Values
# =====================================================
print("\n[2] Missing Values")

missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("✅ No missing values.")
else:
    print(missing)

# =====================================================
# 3. Infinite Values
# =====================================================
print("\n[3] Infinite Values")

numeric_df = df.select_dtypes(include=[np.number])

if np.isinf(numeric_df.values).sum() == 0:
    print("✅ No infinite values.")
else:
    print("❌ Infinite values detected!")

# =====================================================
# 4. Anomaly Statistics
# =====================================================
print("\n[4] Anomaly Statistics")

if "is_anomalous" in df.columns:
    print(df["is_anomalous"].value_counts())

# =====================================================
# 5. Attack Distribution
# =====================================================
print("\n[5] Attack Type Distribution")

if "attack_type" in df.columns:
    print(df["attack_type"].value_counts(dropna=False))

# =====================================================
# 6. Risk Score Statistics
# =====================================================
print("\n[6] Risk Score")

if "risk_score" in df.columns:

    print(df["risk_score"].describe())

    normal = df[df["is_anomalous"] == 0]
    anomalous = df[df["is_anomalous"] == 1]

    print("\nAverage Risk Score")

    print(f"Normal Sessions    : {normal['risk_score'].mean():.2f}")
    print(f"Anomalous Sessions : {anomalous['risk_score'].mean():.2f}")

# =====================================================
# 7. Check Anomaly Metadata
# =====================================================
print("\n[7] Metadata Consistency")

bad_attack = df[
    (df["is_anomalous"] == 1) &
    (
        df["attack_type"].isna() |
        (df["attack_type"] == "None") |
        (df["attack_type"] == "NONE")
    )
]

bad_risk = df[
    (df["is_anomalous"] == 1) &
    (df["risk_score"] <= 0)
]

print(f"Anomalous rows with missing attack_type : {len(bad_attack)}")
print(f"Anomalous rows with zero risk_score     : {len(bad_risk)}")

# =====================================================
# 8. Normal Sessions with High Risk
# =====================================================
print("\n[8] Normal Sessions Having Risk > 0")

normal_high = df[
    (df["is_anomalous"] == 0) &
    (df["risk_score"] > 0)
]

print(f"Count : {len(normal_high)}")

if len(normal_high):
    print(normal_high[[
        "session_id",
        "employee_id",
        "risk_score"
    ]].head(10))

# =====================================================
# 9. Behavioral Deviations
# =====================================================
print("\n[9] Behavioral Deviation Summary")

deviation_cols = [
    "location_deviation",
    "device_deviation",
    "browser_deviation",
    "operating_system_deviation",
    "working_hours_deviation",
    "resource_access_deviation"
]

for col in deviation_cols:

    if col not in df.columns:
        continue

    print("\n", col)
    print(df[col].describe())

    outside = df[(df[col] < 0) | (df[col] > 1)]

    if len(outside) == 0:
        print("✅ All values between 0 and 1")
    else:
        print(f"❌ {len(outside)} values outside [0,1]")

# =====================================================
# 10. Deviation Comparison
# =====================================================
print("\n[10] Average Deviations")

for col in deviation_cols:

    if col not in df.columns:
        continue

    normal_avg = df[df["is_anomalous"] == 0][col].mean()
    anomaly_avg = df[df["is_anomalous"] == 1][col].mean()

    print(f"{col:30s} Normal={normal_avg:.4f}  Anomaly={anomaly_avg:.4f}")

# =====================================================
# 11. Attack vs Risk
# =====================================================
print("\n[11] Average Risk per Attack")

if "attack_type" in df.columns:
    print(
        df.groupby("attack_type")["risk_score"]
        .agg(["count", "mean", "min", "max"])
        .sort_values("mean", ascending=False)
    )

# =====================================================
# 12. Overall Result
# =====================================================
print("\n" + "=" * 80)
print("Validation Complete")
print("=" * 80)