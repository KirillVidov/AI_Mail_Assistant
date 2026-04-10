import pandas as pd

df = pd.read_csv('data/processed/email_reply_pairs.csv')

print("Columns:", df.columns.tolist())
print(f"\nTotal pairs: {len(df)}")
print("\nFirst 3 examples:\n")

for i in range(3):
    print(f"--- PAIR {i+1} ---")
    print(f"Original: {df.iloc[i]['original_body'][:150]}...")
    print(f"Reply: {df.iloc[i]['reply_body'][:150]}...")
    print()