"""
Автоматический классификатор входящих писем (РУССКАЯ ВЕРСИЯ)
Работает в фоновом режиме, проверяет новые письма каждые 60 секунд
"""

import os
import pickle
import re
import time
import torch
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from datetime import datetime

# Импорт архитектуры модели
from architecture import EmailClassifierCNN_LSTM

# Настройки Gmail API
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

# Категории
CATEGORIES = ['work', 'personal', 'spam', 'promo']
LABEL_PREFIX = 'AI/'

print("=" * 60)
print("🇷🇺 Автоматический классификатор русских email")
print("=" * 60)

# Загрузка русской модели
print("\n📦 Загрузка русской модели классификации...")
checkpoint = torch.load('best_model_russian.pth', map_location=torch.device('cpu'))

# Загрузка русского словаря
with open('russian_vocabulary.pkl', 'rb') as f:
    vocab = pickle.load(f)

# Создание модели
vocab_size = checkpoint['model_state_dict']['embedding.weight'].shape[0]
embedding_dim = 128
num_classes = 4

model = EmailClassifierCNN_LSTM(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_classes=num_classes
)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("✓ Русская модель загружена")


def preprocess_text(text):
    """Предобработка русского текста"""
    text = text.lower()
    # Оставляем только кириллицу, латиницу и цифры
    text = re.sub(r'[^а-яёa-z0-9\s]', '', text)
    return text


def text_to_sequence(text, vocab, max_len=100):
    """Конвертация текста в последовательность"""
    words = preprocess_text(text).split()
    sequence = [vocab.get(word, vocab.get('<UNK>', 1)) for word in words]

    if len(sequence) < max_len:
        sequence += [vocab.get('<PAD>', 0)] * (max_len - len(sequence))
    else:
        sequence = sequence[:max_len]

    return torch.tensor([sequence], dtype=torch.long)


def classify_email(text):
    """Классификация email"""
    with torch.no_grad():
        input_seq = text_to_sequence(text, vocab)
        output = model(input_seq)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

        return {
            'category': CATEGORIES[predicted_class],
            'confidence': round(confidence * 100, 2)
        }


def get_gmail_service():
    """Подключение к Gmail API"""
    creds = None

    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return build('gmail', 'v1', credentials=creds)


def get_or_create_label(service, label_name):
    """Получить или создать метку"""
    try:
        # Поиск существующей метки
        results = service.users().labels().list(userId='me').execute()
        labels = results.get('labels', [])

        for label in labels:
            if label['name'] == label_name:
                return label['id']

        # Создание новой метки
        label_object = {
            'name': label_name,
            'labelListVisibility': 'labelShow',
            'messageListVisibility': 'show'
        }
        created_label = service.users().labels().create(userId='me', body=label_object).execute()
        print(f"✓ Создана новая метка: {label_name}")
        return created_label['id']

    except Exception as e:
        print(f"Ошибка при работе с меткой {label_name}: {e}")
        return None


def get_message_text(service, msg_id):
    """Извлечение текста письма"""
    try:
        message = service.users().messages().get(userId='me', id=msg_id, format='full').execute()

        subject = ''
        body = ''

        # Извлечение заголовков
        if 'payload' in message:
            headers = message['payload'].get('headers', [])
            for header in headers:
                if header['name'] == 'Subject':
                    subject = header['value']
                    break

            # Извлечение тела
            if 'parts' in message['payload']:
                for part in message['payload']['parts']:
                    if part['mimeType'] == 'text/plain':
                        if 'data' in part['body']:
                            import base64
                            body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                            break
            elif 'body' in message['payload'] and 'data' in message['payload']['body']:
                import base64
                body = base64.urlsafe_b64decode(message['payload']['body']['data']).decode('utf-8')

        return f"{subject} {body}"

    except Exception as e:
        print(f"Ошибка при извлечении текста письма {msg_id}: {e}")
        return ""


def apply_label(service, msg_id, category):
    """Применение метки к письму"""
    try:
        label_name = f"{LABEL_PREFIX}{category.upper()}"
        label_id = get_or_create_label(service, label_name)

        if label_id:
            service.users().messages().modify(
                userId='me',
                id=msg_id,
                body={'addLabelIds': [label_id]}
            ).execute()
            return True
        return False

    except Exception as e:
        print(f"Ошибка при применении метки: {e}")
        return False


def process_unread_emails(service):
    """Обработка непрочитанных писем"""
    try:
        # Поиск непрочитанных писем без меток AI/*
        query = 'is:unread -label:AI/WORK -label:AI/PERSONAL -label:AI/SPAM -label:AI/PROMO'
        results = service.users().messages().list(userId='me', q=query, maxResults=10).execute()
        messages = results.get('messages', [])

        if not messages:
            return 0

        processed = 0
        for message in messages:
            msg_id = message['id']

            # Извлечение текста
            text = get_message_text(service, msg_id)

            if text.strip():
                # Классификация
                result = classify_email(text)
                category = result['category']
                confidence = result['confidence']

                # Применение метки
                if apply_label(service, msg_id, category):
                    # Ограничение длины темы для вывода
                    subject_preview = text[:50].replace('\n', ' ')
                    print(f"✓ Классифицировано: '{subject_preview}...' как {category.upper()} ({confidence:.1f}%)")
                    processed += 1

        return processed

    except Exception as e:
        print(f"Ошибка при обработке писем: {e}")
        return 0


def main():
    """Основной цикл"""
    print("\n🔐 Подключение к Gmail API...")

    try:
        service = get_gmail_service()
        print("✓ Подключено к Gmail")
    except Exception as e:
        print(f"❌ Ошибка подключения к Gmail: {e}")
        print("\nУбедитесь что:")
        print("1. Файл credentials.json находится в корне проекта")
        print("2. У вас есть доступ к интернету")
        return

    print("\n" + "=" * 60)
    print("🚀 Автоклассификатор запущен!")
    print("=" * 60)
    print("📧 Проверка новых писем каждые 60 секунд...")
    print("🛑 Нажмите Ctrl+C для остановки\n")

    check_count = 0

    try:
        while True:
            check_count += 1
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"[{current_time}] Проверка #{check_count}...", end=" ")

            processed = process_unread_emails(service)

            if processed > 0:
                print(f"Обработано: {processed} писем")
            else:
                print("Новых писем нет")

            time.sleep(60)

    except KeyboardInterrupt:
        print("\n\n🛑 Автоклассификатор остановлен")
        print("=" * 60)


if __name__ == '__main__':
    main()