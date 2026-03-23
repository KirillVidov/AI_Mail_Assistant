import torch
import torch.nn as nn
import torch.nn.functional as F


class EmailClassifierCNN_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128,
                 num_classes=4, dropout=0.5):
        """
        Args:
            vocab_size: размер словаря
            embedding_dim: размерность векторов слов
            hidden_dim: размерность скрытого слоя LSTM
            num_classes: количество категорий (4: work/personal/spam/promo)
            dropout: вероятность dropout для регуляризации
        """
        super(EmailClassifierCNN_LSTM, self).__init__()
        # 1. Embedding Layer
        # Преобразует индексы слов в векторы
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # индекс для PAD токена
        )

        # 2. Convolutional Layers
        self.conv1 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=64,
            kernel_size=3,  # 3-граммы
            padding=1  # (3-1)//2 = 1
        )

        self.conv2 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=64,
            kernel_size=5,  # 5-граммы (изменено с 4 на 5)
            padding=2  # (5-1)//2 = 2
        )

        self.conv3 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=64,
            kernel_size=7,  # 7-граммы (изменено с 5 на 7)
            padding=3  # (7-1)//2 = 3
        )

        # 3. Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=192,  # 64*3 от трёх CNN
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # 4. Attention Layer
        self.attention = AttentionLayer(hidden_dim * 2)

        # 5. Classification Layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)

        # Batch Normalization для стабильности обучения
        self.batch_norm = nn.BatchNorm1d(192)

        self._print_model_info()

    def forward(self, x):
        # x shape: (batch_size, seq_length)

        # 1. Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)

        #(batch, channels, seq_len)
        embedded_t = embedded.permute(0, 2, 1)

        # 2. CNN с разными kernel sizes
        conv_out1 = F.relu(self.conv1(embedded_t))  # (batch, 64, seq_len)
        conv_out2 = F.relu(self.conv2(embedded_t))  # (batch, 64, seq_len)
        conv_out3 = F.relu(self.conv3(embedded_t))  # (batch, 64, seq_len)

        # Объединяем выходы CNN
        conv_out = torch.cat([conv_out1, conv_out2, conv_out3], dim=1)  # (batch, 192, seq_len)

        # Batch Normalization
        conv_out = self.batch_norm(conv_out)

        # Возвращаем форму для LSTM (batch, seq_len, features)
        conv_out = conv_out.permute(0, 2, 1)

        # 3. BiLSTM
        lstm_out, (hidden, cell) = self.lstm(conv_out)
        # lstm_out shape: (batch, seq_len, hidden_dim*2)

        # 4. Attention
        attention_out = self.attention(lstm_out)
        # attention_out shape: (batch, hidden_dim*2)

        # 5. Classification
        out = F.relu(self.fc1(attention_out))
        out = self.dropout(out)
        out = self.fc2(out)  # (batch, num_classes)

        return out

    def _print_model_info(self):
        """Вывод информации о модели"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # Вычисляем веса внимания для каждого слова
        attention_weights = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)

        # Взвешенная сумма
        weighted_output = torch.sum(attention_weights * lstm_output, dim=1)
        # (batch, hidden_dim)

        return weighted_output


def test_model():
    """Тестирование модели"""
    # Параметры
    vocab_size = 172  # из нашего словаря
    batch_size = 4
    seq_length = 128

    # Создаём модель
    model = EmailClassifierCNN_LSTM(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_classes=4,
        dropout=0.5
    )

    # Тестовые данные
    test_input = torch.randint(0, vocab_size, (batch_size, seq_length))

    # Прямой проход
    model.eval()
    with torch.no_grad():
        output = model(test_input)

    # Применяем softmax для получения вероятностей
    probabilities = F.softmax(output, dim=1)
    print(f"\nВероятности для первого примера:")
    categories = ['Работа', 'Личное', 'Спам', 'Промо']
    for i, category in enumerate(categories):
        print(f"  {category}: {probabilities[0][i].item():.4f}")

    return model


if __name__ == "__main__":
    model = test_model()