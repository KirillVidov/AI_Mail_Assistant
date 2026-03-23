import re
import pickle
from collections import Counter
import pandas as pd


class EmailVocabulary:
    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        # Специальные токены
        self.PAD_TOKEN = '<PAD>'  # Заполнитель для коротких текстов
        self.UNK_TOKEN = '<UNK>'  # Неизвестное слово
        self.PAD_IDX = 0
        self.UNK_IDX = 1

        # Словари для преобразования слово <-> индекс
        self.word2idx = {self.PAD_TOKEN: self.PAD_IDX, self.UNK_TOKEN: self.UNK_IDX}
        self.idx2word = {self.PAD_IDX: self.PAD_TOKEN, self.UNK_IDX: self.UNK_TOKEN}

        # Статистика по словам
        self.word_counts = Counter()

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""

        #нижнему регистру
        text = text.lower()

        #email адреса
        text = re.sub(r'\S+@\S+', ' ', text)

        #URL
        text = re.sub(r'http\S+|www.\S+', ' ', text)

        #числа
        # text = re.sub(r'\d+', ' ', text)

        #буквы, цифры и пробелы
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

        #множественные пробелы
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def tokenize(self, text):
        text = self.clean_text(text)
        tokens = text.split()
        return tokens

    def build_vocabulary(self, texts, min_freq=2):
        print(len(texts))
        # Подсчёт частоты слов
        for i, text in enumerate(texts):
            tokens = self.tokenize(text)
            self.word_counts.update(tokens)

        # Фильтруем редкие слова
        filtered_words = {word: count for word, count in self.word_counts.items()
                          if count >= min_freq}

        #самые частые слова
        most_common = Counter(filtered_words).most_common(self.max_vocab_size - 2)

        # Заполняем словари
        for idx, (word, count) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        #самые частые слова
        print(f"\nТоп-20 самых частых слов:")
        for word, count in most_common[:20]:
            print(f"  {word}: {count}")

    def encode(self, text, max_length=128):
        tokens = self.tokenize(text)

        # Преобразуем слова в индексы
        indices = [self.word2idx.get(token, self.UNK_IDX) for token in tokens]

        # Padding (дополнение до max_length) или обрезка
        if len(indices) < max_length:
            # Дополняем нулями (PAD)
            indices += [self.PAD_IDX] * (max_length - len(indices))
        else:
            # Обрезаем до max_length
            indices = indices[:max_length]

        return indices

    def decode(self, indices):
        words = [self.idx2word.get(idx, self.UNK_TOKEN) for idx in indices]
        # Убираем PAD токены
        words = [word for word in words if word != self.PAD_TOKEN]
        return ' '.join(words)

    def save(self, filepath):
        data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_counts': self.word_counts,
            'max_vocab_size': self.max_vocab_size
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']
        self.word_counts = data['word_counts']
        self.max_vocab_size = data['max_vocab_size']

        print(f"Словарь загружен (размер: {len(self.word2idx)} слов)")

    def get_stats(self):
        return {
            'vocab_size': len(self.word2idx),
            'total_words': sum(self.word_counts.values()),
            'unique_words': len(self.word_counts),
            'most_common': self.word_counts.most_common(10)
        }


def test_vocabulary():
    df = pd.read_csv('./data/processed/test_emails.csv')

    # Объединяем тему и тело письма
    df['full_text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')

    # Создаём словарь
    vocab = EmailVocabulary(max_vocab_size=5000)
    vocab.build_vocabulary(df['full_text'].tolist(), min_freq=2)

    # Сохраняем
    vocab.save('./data/processed/vocabulary.pkl')

    # Тест
    test_text = "Meeting scheduled for project review tomorrow"
    print(f"\nИсходный текст: {test_text}")

    encoded = vocab.encode(test_text, max_length=20)
    print(f"Закодированный: {encoded}")

    decoded = vocab.decode(encoded)
    print(f"Декодированный: {decoded}")
    stats = vocab.get_stats()

    return vocab


if __name__ == "__main__":
    vocab = test_vocabulary()