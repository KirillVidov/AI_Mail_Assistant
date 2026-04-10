"""
Балансировка датасета для улучшения классификации
Приводит все категории к одинаковому количеству примеров
"""

import pandas as pd
import numpy as np
from sklearn.utils import resample

print("="*60)
print("БАЛАНСИРОВКА ДАТАСЕТА ДЛЯ КЛАССИФИКАЦИИ")
print("="*60)

# Загрузка данных
print("\n1. Загрузка данных...")
df = pd.read_csv('data/processed/labeled_train.csv')

print(f"Всего писем: {len(df)}")
print("\nТекущее распределение:")
print(df['category_id'].value_counts().sort_index())

# Разделяем по категориям
df_work = df[df['category_id'] == 0]
df_personal = df[df['category_id'] == 1]
df_spam = df[df['category_id'] == 2]
df_promo = df[df['category_id'] == 3]

print(f"\nWORK: {len(df_work)} писем")
print(f"PERSONAL: {len(df_personal)} писем")
print(f"SPAM: {len(df_spam)} писем")
print(f"PROMO: {len(df_promo)} писем")

# Целевое количество для каждой категории
target_samples = 5000

print(f"\n2. Балансировка до {target_samples} писем на категорию...")

# Undersampling для WORK (уменьшаем с 5394 до 3000)
df_work_balanced = resample(df_work,
                            replace=False,
                            n_samples=target_samples,
                            random_state=42)
print(f"✓ WORK: {len(df_work)} → {len(df_work_balanced)}")

# Oversampling для остальных (увеличиваем с ~1796 до 3000)
df_personal_balanced = resample(df_personal,
                               replace=True,
                               n_samples=target_samples,
                               random_state=42)
print(f"✓ PERSONAL: {len(df_personal)} → {len(df_personal_balanced)}")

df_spam_balanced = resample(df_spam,
                            replace=True,
                            n_samples=target_samples,
                            random_state=42)
print(f"✓ SPAM: {len(df_spam)} → {len(df_spam_balanced)}")

df_promo_balanced = resample(df_promo,
                             replace=True,
                             n_samples=target_samples,
                             random_state=42)
print(f"✓ PROMO: {len(df_promo)} → {len(df_promo_balanced)}")

# Объединяем обратно
df_balanced = pd.concat([
    df_work_balanced,
    df_personal_balanced,
    df_spam_balanced,
    df_promo_balanced
])

# Перемешиваем
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("\n3. Результат балансировки:")
print(f"Всего писем: {len(df_balanced)}")
print("\nНовое распределение:")
print(df_balanced['category_id'].value_counts().sort_index())

print("\nПроцентное распределение:")
percentages = (df_balanced['category_id'].value_counts().sort_index() / len(df_balanced) * 100)
for cat_id, pct in percentages.items():
    category_names = {0: 'WORK', 1: 'PERSONAL', 2: 'SPAM', 3: 'PROMO'}
    print(f"{category_names[cat_id]}: {pct:.1f}%")

# Сохранение
output_path = 'data/processed/labeled_train_balanced.csv'
df_balanced.to_csv(output_path, index=False)
print(f"\n4. Сбалансированный датасет сохранен: {output_path}")

