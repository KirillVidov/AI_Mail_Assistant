import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class Trainer:
    def __init__(self, model, train_loader, val_loader, device, learning_rate=0.001):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Функция потерь - CrossEntropyLoss для классификации
        self.criterion = nn.CrossEntropyLoss()

        # Оптимизатор - Adam (лучше чем обычный SGD)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Scheduler - уменьшаю learning rate если застреваю
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
        )

        # История обучения
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def train_epoch(self):
        self.model.train()  # Режим обучения (включает dropout и т.д.)

        total_loss = 0
        correct = 0
        total = 0

        # tqdm - это прогресс-бар, чтобы видеть процесс
        progress_bar = tqdm(self.train_loader, desc='Training')

        for batch in progress_bar:
            # Получаю данные из батча
            inputs = batch['input'].to(self.device)
            labels = batch['label'].to(self.device)

            # Обнуляю градиенты (важно!)
            self.optimizer.zero_grad()

            # Forward pass - прогоняю данные через модель
            outputs = self.model(inputs)

            # Считаю ошибку
            loss = self.criterion(outputs, labels)

            # Backward pass - считаю градиенты
            loss.backward()

            # Gradient clipping - чтобы градиенты не взорвались
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Обновляю веса модели
            self.optimizer.step()

            # Статистика для этого батча
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Обновляю прогресс-бар
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })

        # Средние значения за эпоху
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy

    def validate(self):
        self.model.eval()  # Режим оценки (выключает dropout)

        total_loss = 0
        correct = 0
        total = 0

        all_predictions = []
        all_labels = []

        # torch.no_grad() - не считаю градиенты (экономит память)
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                inputs = batch['input'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Статистика
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Сохраняю для confusion matrix
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total

        return avg_loss, accuracy, all_predictions, all_labels

    def train(self, epochs=20, patience=5):
        best_val_acc = 0
        patience_counter = 0

        for epoch in range(epochs):
            print(f"\nЭпоха {epoch + 1}/{epochs}")
            print("-" * 60)

            # Обучаю одну эпоху
            train_loss, train_acc = self.train_epoch()

            # Валидирую
            val_loss, val_acc, predictions, labels = self.validate()

            # Сохраняю в историю
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Вывожу результаты
            print(f"\nРезультаты:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

            # Обновляю learning rate если нужно
            self.scheduler.step(val_loss)

            # Early stopping - проверяю улучшилась ли модель
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0

                # Сохраняю лучшую модель
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'history': self.history
                }, 'best_model.pth')

                print(f"  ✓ Новая лучшая модель! Точность: {val_acc:.2f}%")
            else:
                patience_counter += 1
                print(f"  Без улучшений ({patience_counter}/{patience})")

            # Останавливаюсь если давно не было улучшений
            if patience_counter >= patience:
                print(f"\nEarly stopping после эпохи {epoch + 1}")
                print(f"Лучшая точность: {best_val_acc:.2f}%")
                break

        print(f"Лучшая точность на валидации: {best_val_acc:.2f}%")

        return self.history


def plot_training_history(history, save_path='training_curves.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # График потерь
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', marker='o')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', marker='s')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Loss')
    ax1.set_title('Потери модели')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График точности
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', marker='o')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', marker='s')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Точность модели')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Графики сохранены: {save_path}")
    plt.close()


def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            inputs = batch['input'].to(device)
            labels = batch['label'].to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"\nТочность на тестовых данных: {accuracy:.2f}%")

    return np.array(all_predictions), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred, categories, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories, yticklabels=categories)
    plt.title('Матрица ошибок', fontsize=16)
    plt.ylabel('Истинная метка')
    plt.xlabel('Предсказанная метка')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Подготавливаю данные
    from step3_1_prepare_data import prepare_data

    train_loader, val_loader, test_loader, vocab_size = prepare_data(
        csv_path='./data/processed/test_emails.csv',
        vocab_path='./data/processed/vocabulary.pkl',
        batch_size=16,
        max_length=128
    )

    # Создаю модель
    from step2_2_model_architecture import EmailClassifierCNN_LSTM

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nИспользую устройство: {device}")

    model = EmailClassifierCNN_LSTM(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_classes=4,
        dropout=0.5
    )

    # Создаю тренер и обучаю
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=0.001
    )

    history = trainer.train(epochs=20, patience=5)

    # Строю графики
    plot_training_history(history)

    # Загружаю лучшую модель для тестирования
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Тестирую
    predictions, true_labels = evaluate_model(model, test_loader, device)

    # Отчёт по классификации
    categories = ['Работа', 'Личное', 'Спам', 'Промо']
    print(classification_report(true_labels, predictions, target_names=categories))

    # Матрица ошибок
    plot_confusion_matrix(true_labels, predictions, categories)

if __name__ == "__main__":
    main()