import pandas as pd
import os

def download_with_auth():
    # Устанавливаю библиотеку
    try:
        from datasets import load_dataset
    except ImportError:
        import subprocess
        subprocess.check_call(['pip', 'install', 'datasets', '--break-system-packages'])
        from datasets import load_dataset

    # Проверяю есть ли токен
    import os
    token = os.environ.get('HF_TOKEN')

    if not token:
        token = input("Token: ").strip()
        if not token:
            return None, None

    try:
        dataset = load_dataset(
            "jason23322/high-accuracy-email-classifier",
            token=token  # Передаю токен для авторизации
        )

        # Показываю структуру
        print(f"\nСтруктура:")
        for split_name, split_data in dataset.items():
            print(f"  {split_name}: {len(split_data)} примеров")

        # Показываю пример
        print(f"\nПример письма:")
        example = dataset['train'][0]
        for key, value in example.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}...")
            else:
                print(f"  {key}: {value}")

        # Преобразую в DataFrame
        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test']) if 'test' in dataset else None

        if test_df is None:
            # Если нет test, создаю из train
            from sklearn.model_selection import train_test_split
            train_df, test_df = train_test_split(
                train_df, test_size=0.2, random_state=42
            )

        print(f"\nTrain: {len(train_df)} примеров")
        print(f"Test: {len(test_df)} примеров")

        return train_df, test_df

    except Exception as e:
        return None, None


def map_categories(train_df, test_df):
    label_col = 'category' if 'category' in train_df.columns else 'label'

    # Маппинг
    category_mapping = {
        'forum': 'work',
        'promotions': 'promo',
        'social_media': 'personal',
        'spam': 'spam',
        'updates': 'work',
        'verify_code': 'work',
        # На всякий случай разные варианты написания
        'Forum': 'work',
        'Promotions': 'promo',
        'Social Media': 'personal',
        'Social': 'personal',
        'Spam': 'spam',
        'Updates': 'work',
        'Verify Code': 'work',
        'social': 'personal',
        'promotion': 'promo',
    }

    my_category_ids = {'work': 0, 'personal': 1, 'spam': 2, 'promo': 3}

    # Применяю маппинг
    train_df['category'] = train_df[label_col].map(category_mapping)
    train_df['category_id'] = train_df['category'].map(my_category_ids)

    test_df['category'] = test_df[label_col].map(category_mapping)
    test_df['category_id'] = test_df['category'].map(my_category_ids)

    # Удаляю строки где маппинг не сработал
    train_df = train_df.dropna(subset=['category'])
    test_df = test_df.dropna(subset=['category'])

    print(f"\nМои категории после маппинга:")
    print(train_df['category'].value_counts())

    print(f"\nРаспределение по category_id:")
    for cat_id, cat_name in my_category_ids.items():
        count = (train_df['category'] == cat_name).sum()
        print(f"  {cat_id} ({cat_name}): {count} писем")

    return train_df, test_df


def save_dataset(train_df, test_df):
    os.makedirs('./data/processed', exist_ok=True)

    train_path = './data/processed/labeled_train.csv'
    test_path = './data/processed/labeled_test.csv'

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Показываю примеры
    for cat in ['work', 'personal', 'spam', 'promo']:
        samples = train_df[train_df['category'] == cat].head(1)
        if len(samples) > 0:
            sample = samples.iloc[0]
            print(f"\n📧 {cat.upper()}:")

            if 'subject' in sample:
                print(f"   Тема: {sample['subject'][:70]}...")
            if 'body' in sample:
                print(f"   Тело: {str(sample['body'])[:80]}...")
            elif 'text' in sample:
                print(f"   Текст: {str(sample['text'])[:80]}...")


def main():
    # Загружаю с авторизацией
    train_df, test_df = download_with_auth()

    # Маппинг категорий
    train_df, test_df = map_categories(train_df, test_df)

    # Сохраняю
    save_dataset(train_df, test_df)


if __name__ == "__main__":
    main()