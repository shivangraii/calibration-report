"""Step 1: validate the raw CSV and create the processed question file."""
import pandas as pd
import config

df = pd.read_csv(config.RAW_CSV)
required = {"id", "question", "answer", "category", "difficulty"}
missing = required - set(df.columns)
assert not missing, f"Missing columns: {missing}"

for c in ["id", "question", "answer", "category", "difficulty"]:
    df[c] = df[c].astype(str).str.strip()

print(f"Rows: {len(df)}")
print("Missing values:\n", df[list(required)].replace("", pd.NA).isna().sum().to_string())
print("Duplicate ids:", df["id"].duplicated().sum())
print("Duplicate questions:", df["question"].duplicated().sum())

df = df.drop_duplicates(subset="id").reset_index(drop=True)

# Merge tiny categories / difficulty levels so per-group ECE is not pure noise.
counts = df["category"].value_counts()
small = counts[counts < config.MIN_GROUP_N].index
df["category_merged"] = df["category"].where(~df["category"].isin(small), "Other (merged)")
df["difficulty_merged"] = df["difficulty"].replace({"Very Hard": "Hard+", "Hard": "Hard+"})

df = df.rename(columns={"answer": "reference"})
config.PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(config.PROCESSED_CSV, index=False)

print("\nCategories after merging:\n", df["category_merged"].value_counts().to_string())
print("\nDifficulty after merging:\n", df["difficulty_merged"].value_counts().to_string())
print(f"\nSaved -> {config.PROCESSED_CSV}")
