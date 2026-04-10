import pandas as pd

# Загрузка обучающего датасета
df = pd.read_csv('data/processed/labeled_train.csv')

# Посмотреть все колонки
print("Колонки в датасете:")
print(df.columns.tolist())

print("\n" + "="*50)
print("Распределение классов:")
print(df['category_id'].value_counts().sort_index())

print("\n" + "="*50)
print("Процентное распределение:")
counts = df['category_id'].value_counts().sort_index()
percentages = (counts / counts.sum() * 100).round(2)
for cat_id, pct in percentages.items():
    print(f"Категория {cat_id}: {pct}%")

# Mapping категорий
category_mapping = {
    0: 'work',
    1: 'personal',
    2: 'spam',
    3: 'promo'
}

print("\n" + "="*50)
print("Детальное распределение:")
for cat_id, count in counts.items():
    pct = (count / counts.sum() * 100)
    print(f"{category_mapping.get(cat_id, 'unknown')}: {count} писем ({pct:.1f}%)")