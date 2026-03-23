import torch
import pandas as pd
import pickle
from tqdm import tqdm
import re


def classify_enron_emails():
    try:
        enron_df = pd.read_csv('./data/processed/enron_emails.csv')
    except FileNotFoundError:
        # Пробую тестовый
        try:
            enron_df = pd.read_csv('./data/processed/test_emails.csv')
        except:
            return

    try:
        checkpoint = torch.load('best_model.pth', map_location='cpu')
        print(f"  Точность на валидации: {checkpoint['val_acc']:.2f}%")
    except FileNotFoundError:
        return

    # Загружаю словарь
    try:
        with open('./data/processed/transfer_vocabulary.pkl', 'rb') as f:
            vocab_data = pickle.load(f)
        print(f"Словарь загружен (размер: {len(vocab_data['word2idx'])})")
    except FileNotFoundError:
        return

    # Создаю модель
    from architecture import EmailClassifierCNN_LSTM

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = EmailClassifierCNN_LSTM(
        vocab_size=len(vocab_data['word2idx']),
        embedding_dim=128,
        hidden_dim=128,
        num_classes=4,
        dropout=0.5
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Объединяю subject и body
    if 'subject' in enron_df.columns and 'body' in enron_df.columns:
        texts = (enron_df['subject'].fillna('') + ' ' + enron_df['body'].fillna('')).tolist()
    elif 'subject' in enron_df.columns:
        texts = enron_df['subject'].fillna('').tolist()
    else:
        return

    print(f"Подготовлено {len(texts)} текстов")

    # Функция кодирования
    def encode_text(text, word2idx, max_length=128):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()

        indices = [word2idx.get(token, 1) for token in tokens]  # 1 = UNK

        if len(indices) < max_length:
            indices += [0] * (max_length - len(indices))  # 0 = PAD
        else:
            indices = indices[:max_length]

        return indices

    # Классифицирую батчами
    batch_size = 32
    all_predictions = []
    all_confidences = []

    print(f"Классифицирую {len(texts)} писем (batch_size={batch_size})...")

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc='Классификация'):
            batch_texts = texts[i:i + batch_size]

            # Кодирую батч
            batch_encoded = [encode_text(text, vocab_data['word2idx']) for text in batch_texts]
            batch_tensor = torch.tensor(batch_encoded, dtype=torch.long).to(device)

            # Предсказываю
            outputs = model(batch_tensor)
            probabilities = torch.softmax(outputs, dim=1)

            # Получаю предсказания и уверенности
            confidences, predictions = torch.max(probabilities, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())

    category_map = {0: 'work', 1: 'personal', 2: 'spam', 3: 'promo'}

    enron_df['category_id'] = all_predictions
    enron_df['category'] = enron_df['category_id'].map(category_map)
    enron_df['confidence'] = all_confidences

    # Сохраняю
    output_path = './data/processed/enron_emails_classified.csv'
    enron_df.to_csv(output_path, index=False)

    print(f"Результаты сохранены: {output_path}")

    print(f"\nРаспределение по категориям:")
    for cat_id in range(4):
        cat_name = category_map[cat_id]
        count = (enron_df['category_id'] == cat_id).sum()
        percent = count / len(enron_df) * 100
        avg_conf = enron_df[enron_df['category_id'] == cat_id]['confidence'].mean()
        print(f"  {cat_name.capitalize():10}: {count:4} писем ({percent:5.1f}%) | Уверенность: {avg_conf:.3f}")

    print(f"\nСредняя уверенность модели: {enron_df['confidence'].mean():.3f}")

    for cat in ['work', 'personal', 'spam', 'promo']:
        # Беру пример с высокой уверенностью
        cat_emails = enron_df[enron_df['category'] == cat].nlargest(1, 'confidence')

        if len(cat_emails) > 0:
            email = cat_emails.iloc[0]
            print(f"\n📧 {cat.upper()} (уверенность: {email['confidence']:.3f})")
            if 'from' in email:
                print(f"   От: {email['from']}")
            if 'subject' in email:
                print(f"   Тема: {email['subject'][:60]}...")
            if 'body' in email:
                print(f"   Тело: {str(email['body'])[:80]}...")
            print("-" * 60)

    # Примеры низкой уверенности (сомнительные)
    print(f"\nПисьма с низкой уверенностью (< 0.6):")
    uncertain = enron_df[enron_df['confidence'] < 0.6]
    print(f"Найдено {len(uncertain)} писем ({len(uncertain) / len(enron_df) * 100:.1f}%)")

    if len(uncertain) > 0:
        print(f"\nПример:")
        sample = uncertain.iloc[0]
        print(f"  Категория: {sample['category']} (уверенность: {sample['confidence']:.3f})")
        if 'subject' in sample:
            print(f"  Тема: {sample['subject'][:60]}...")

if __name__ == "__main__":
    classify_enron_emails()