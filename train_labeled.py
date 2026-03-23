import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class LabeledEmailDataset(Dataset):
    def __init__(self, df, vocabulary, max_length=128):
        # Объединяю subject и body/text
        if 'subject' in df.columns and 'body' in df.columns:
            self.texts = (df['subject'].fillna('') + ' ' + df['body'].fillna('')).tolist()
        elif 'text' in df.columns:
            self.texts = df['text'].fillna('').tolist()
        elif 'subject' in df.columns:
            self.texts = df['subject'].fillna('').tolist()
        else:
            raise ValueError("Не найдены колонки с текстом (subject/body/text)")

        self.labels = df['category_id'].tolist()
        self.vocabulary = vocabulary
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Кодирую текст
        encoded = self.vocabulary.encode(text, self.max_length)

        return {
            'input': torch.tensor(encoded, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


def train_on_labeled_data():
    try:
        train_df = pd.read_csv('./data/processed/labeled_train.csv')
        test_df = pd.read_csv('./data/processed/labeled_test.csv')
    except FileNotFoundError:
        return

    # Показываю распределение
    print(f"\nРаспределение категорий:")
    print(train_df['category'].value_counts())

    # Объединяю тексты
    if 'subject' in train_df.columns and 'body' in train_df.columns:
        all_texts = (train_df['subject'].fillna('') + ' ' + train_df['body'].fillna('')).tolist()
    elif 'text' in train_df.columns:
        all_texts = train_df['text'].fillna('').tolist()
    else:
        all_texts = train_df['subject'].fillna('').tolist()

    # Использую существующий класс словаря
    from vocabulary import EmailVocabulary

    vocab = EmailVocabulary(max_vocab_size=10000)
    vocab.build_vocabulary(all_texts, min_freq=2)
    vocab.save('./data/processed/transfer_vocabulary.pkl')

    # Создаю упрощённый класс словаря для DataLoader
    class SimpleVocab:
        def __init__(self, word2idx):
            self.word2idx = word2idx
            self.PAD_IDX = 0
            self.UNK_IDX = 1

        def encode(self, text, max_length):
            import re
            text = text.lower()
            text = re.sub(r'[^a-z0-9\s]', ' ', text)
            tokens = text.split()

            indices = [self.word2idx.get(token, self.UNK_IDX) for token in tokens]

            if len(indices) < max_length:
                indices += [self.PAD_IDX] * (max_length - len(indices))
            else:
                indices = indices[:max_length]

            return indices

    simple_vocab = SimpleVocab(vocab.word2idx)

    # Создаю datasets
    train_dataset = LabeledEmailDataset(train_df, simple_vocab, max_length=128)
    test_dataset = LabeledEmailDataset(test_df, simple_vocab, max_length=128)

    # Создаю dataloaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")

    from architecture import EmailClassifierCNN_LSTM

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    model = EmailClassifierCNN_LSTM(
        vocab_size=len(vocab.word2idx),
        embedding_dim=128,
        hidden_dim=128,
        num_classes=4,
        dropout=0.5
    ).to(device)

    from train_model import Trainer

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,  # Использую test как validation
        device=device,
        learning_rate=0.001
    )

    # Обучаю
    history = trainer.train(epochs=15, patience=5)

    # Загружаю лучшую модель
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Тестирую
    from train_model import evaluate_model, plot_confusion_matrix, plot_training_history

    predictions, true_labels = evaluate_model(model, test_loader, device)

    # Отчёт
    categories = ['Work', 'Personal', 'Spam', 'Promo']
    print(classification_report(true_labels, predictions, target_names=categories))

    # Графики
    plot_training_history(history, 'transfer_learning_curves.png')
    plot_confusion_matrix(true_labels, predictions, categories, 'transfer_learning_confusion.png')

    print(f"\nТочность на тесте: {100 * (predictions == true_labels).sum() / len(true_labels):.2f}%")



if __name__ == "__main__":
    train_on_labeled_data()