"""
Flask API для Email Assistant (РУССКАЯ ВЕРСИЯ)
Запускать: python flask_api_russian.py
API будет доступен на http://localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import pickle
import re
import os
from dotenv import load_dotenv
import requests

# Загрузка переменных окружения из .env файла
load_dotenv()

app = Flask(__name__)
CORS(app)

# Загрузка модели классификации
print("Загрузка русской модели классификации...")

# Импорт класса модели
from architecture import EmailClassifierCNN_LSTM

# Загрузка checkpoint РУССКОЙ модели
checkpoint = torch.load('best_model_russian.pth', map_location=torch.device('cpu'))

# Загрузка РУССКОГО словаря
with open('russian_vocabulary.pkl', 'rb') as f:
    vocab = pickle.load(f)

# Создание модели с правильными параметрами
vocab_size = checkpoint['model_state_dict']['embedding.weight'].shape[0]
embedding_dim = 128
num_classes = 4

classifier_model = EmailClassifierCNN_LSTM(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_classes=num_classes
)

# Загрузка весов
classifier_model.load_state_dict(checkpoint['model_state_dict'])
classifier_model.eval()

print("✓ Русская модель классификации загружена")

# Инициализация HuggingFace API с РУССКОЙ моделью T5
print("Инициализация HuggingFace Inference API (русская модель)...")

# Токен берется из переменной окружения
HF_API_TOKEN = os.getenv('HF_API_TOKEN')

if not HF_API_TOKEN or HF_API_TOKEN == 'YOUR_HF_TOKEN':
    print("⚠️  ВНИМАНИЕ: HuggingFace токен не найден!")
    print("   Создайте файл .env и добавьте: HF_API_TOKEN=ваш_токен")
    print("   Или установите переменную окружения HF_API_TOKEN")

# РУССКАЯ модель T5 для перефразирования
HF_API_URL = "https://api-inference.huggingface.co/models/cointegrated/rut5-base-paraphraser"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

print("✓ HuggingFace API готов к работе (русская T5)")

CATEGORIES = ['work', 'personal', 'spam', 'promo']


def preprocess_text(text):
    """Предобработка русского текста"""
    text = text.lower()
    # Оставляем только кириллицу, латиницу и цифры
    text = re.sub(r'[^а-яёa-z0-9\s]', '', text)
    return text


def text_to_sequence(text, vocab, max_len=100):
    """Конвертация текста в последовательность индексов"""
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
        output = classifier_model(input_seq)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

        return {
            'category': CATEGORIES[predicted_class],
            'confidence': round(confidence * 100, 2),
            'all_probabilities': {
                cat: round(prob.item() * 100, 2)
                for cat, prob in zip(CATEGORIES, probabilities[0])
            }
        }


def normalize_russian_slang(text):
    """Нормализация русского интернет-сленга"""
    slang_map = {
        # Сокращения
        r'\bспс\b': 'спасибо',
        r'\bпжл\b': 'пожалуйста',
        r'\bпжлст\b': 'пожалуйста',
        r'\bпжл\b': 'пожалуйста',
        r'\bзавтра\b': 'завтра',
        r'\bсейчас\b': 'сейчас',
        r'\bщас\b': 'сейчас',
        r'\bчо\b': 'что',
        r'\bчё\b': 'что',
        r'\bшо\b': 'что',
        r'\bток\b': 'только',
        r'\bтя\b': 'тебя',
        r'\bмб\b': 'может быть',
        r'\bхз\b': 'не знаю',
        r'\bвопч\b': 'вообще',
        r'\bнорм\b': 'нормально',
        r'\bок\b': 'хорошо',
        r'\bпон\b': 'понятно',
        r'\bдд\b': 'да да',
        r'\bпрост\b': 'просто',
        r'\bнада\b': 'надо',
        r'\bнадо\b': 'надо',
    }

    normalized = text
    for pattern, replacement in slang_map.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    return normalized


def rephrase_text(text, category='work', max_length=128):
    """Перефразирование текста с помощью русской HuggingFace T5 API"""

    try:
        # Нормализация сленга
        normalized = normalize_russian_slang(text)

        # Русская модель ruT5 работает немного по-другому
        # Она ожидает просто текст без промпта "paraphrase:"

        # Запрос к API
        payload = {
            "inputs": normalized,
            "parameters": {
                "max_length": max_length,
                "temperature": 0.7,
                "num_beams": 5,
                "do_sample": False
            }
        }

        response = requests.post(
            HF_API_URL,
            headers=HF_HEADERS,
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()

            if isinstance(result, list) and len(result) > 0:
                # ruT5 возвращает список с generated_text
                paraphrased = result[0].get('generated_text', normalized)
                return paraphrased.strip()
            elif isinstance(result, dict):
                # Или может вернуть dict
                if 'generated_text' in result:
                    return result['generated_text'].strip()
                elif 'translation_text' in result:
                    return result['translation_text'].strip()
                else:
                    print(f"Неожиданный формат: {result}")
                    return normalized
            else:
                print(f"Неожиданный формат: {result}")
                return normalized

        elif response.status_code == 503:
            print("Модель загружается, подождите 20 сек...")
            return normalized

        else:
            print(f"Ошибка HF API: {response.status_code} - {response.text}")
            return normalized

    except Exception as e:
        print(f"Ошибка: {e}")
        return text


def generate_greeting(recipient_name=None, category='work'):
    """Генерация обращения на русском"""
    if category == 'work':
        if recipient_name:
            return f"Уважаемый {recipient_name},"
        return "Уважаемый коллега,"
    elif category == 'personal':
        if recipient_name:
            return f"Привет, {recipient_name}!"
        return "Привет!"
    else:
        return ""


def generate_signature(sender_name=None, category='work'):
    """Генерация подписи на русском"""
    if category == 'work':
        return f"\n\nС уважением,\n{sender_name or 'Пользователь'}"
    elif category == 'personal':
        return f"\n\n{sender_name or 'Я'}"
    else:
        return ""


# === API ENDPOINTS ===

@app.route('/health', methods=['GET'])
def health():
    """Проверка работоспособности API"""
    return jsonify({'status': 'ok', 'message': 'Russian Email Assistant API is running'})


@app.route('/classify', methods=['POST'])
def classify():
    """Классификация email"""
    try:
        data = request.json
        text = data.get('text', '')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        result = classify_email(text)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/rephrase', methods=['POST'])
def rephrase():
    """Перефразирование текста"""
    try:
        data = request.json
        text = data.get('text', '')
        category = data.get('category', 'work')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        rephrased = rephrase_text(text, category)

        return jsonify({
            'original': text,
            'rephrased': rephrased,
            'category': category
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/compose', methods=['POST'])
def compose():
    """Полная композиция письма"""
    try:
        data = request.json
        draft = data.get('draft', '')
        recipient_name = data.get('recipient_name')
        sender_name = data.get('sender_name', 'Пользователь')

        if not draft:
            return jsonify({'error': 'No draft provided'}), 400

        # 1. Классификация
        classification = classify_email(draft)
        category = classification['category']

        # 2. Перефразирование
        rephrased = rephrase_text(draft, category)

        # 3. Добавление обращения и подписи
        greeting = generate_greeting(recipient_name, category)
        signature = generate_signature(sender_name, category)

        # 4. Полное письмо
        full_email = f"{greeting}\n\n{rephrased}{signature}"

        return jsonify({
            'draft': draft,
            'category': category,
            'confidence': classification['confidence'],
            'greeting': greeting,
            'body': rephrased,
            'signature': signature,
            'full_email': full_email
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/process_email', methods=['POST'])
def process_email():
    """Обработка входящего email"""
    try:
        data = request.json
        subject = data.get('subject', '')
        body = data.get('body', '')

        full_text = f"{subject} {body}"

        if not full_text.strip():
            return jsonify({'error': 'No email content provided'}), 400

        classification = classify_email(full_text)

        return jsonify({
            'category': classification['category'],
            'confidence': classification['confidence'],
            'suggested_label': classification['category'].upper(),
            'action': f"Apply '{classification['category']}' label",
            'all_probabilities': classification['all_probabilities']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🇷🇺 Russian Email Assistant API Server")
    print("=" * 50)
    print("\nEndpoints:")
    print("  GET  /health          - Check API status")
    print("  POST /classify        - Classify email text")
    print("  POST /rephrase        - Rephrase draft text")
    print("  POST /compose         - Full email composition")
    print("  POST /process_email   - Process incoming email")
    print("\nServer running on http://localhost:5000")
    print("=" * 50 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=True)