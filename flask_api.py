"""
Flask API для Email Assistant
Запускать: python flask_api.py
API будет доступен на http://localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import pickle
import re

app = Flask(__name__)
CORS(app)  # Разрешить запросы из Gmail Add-on

# Загрузка модели классификации
print("Загрузка модели классификации...")

# Импорт класса модели из твоего файла
from architecture import EmailClassifierCNN_LSTM

# Загрузка checkpoint
checkpoint = torch.load('best_model.pth', map_location=torch.device('cpu'))

# Загрузка словаря
with open('transfer_vocabulary.pkl', 'rb') as f:
    vocab = pickle.load(f)

# Создание модели с правильными параметрами
# Получаем vocab_size из checkpoint (размер embedding.weight)
vocab_size = checkpoint['model_state_dict']['embedding.weight'].shape[0]
embedding_dim = 128
num_classes = 4

classifier_model = EmailClassifierCNN_LSTM(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_classes=num_classes
)

# Загрузка весов
if isinstance(checkpoint, dict):
    classifier_model.load_state_dict(checkpoint['model_state_dict'])
else:
    classifier_model = checkpoint

classifier_model.eval()

# Инициализация HuggingFace API
print("Инициализация HuggingFace Inference API...")
import requests

# ВАЖНО: Замени 'YOUR_HF_TOKEN' на свой токен от HuggingFace
HF_API_TOKEN = 'YOUR_HF_TOKEN'  # <-- ВСТАВЬ СЮДА СВОЙ ТОКЕН

# Используем T5 paraphraser через новый API
HF_API_URL = "https://api-inference.huggingface.co/models/humarin/chatgpt_paraphraser_on_T5_base"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

print("✓ HuggingFace API готов к работе")

CATEGORIES = ['work', 'personal', 'spam', 'promo']


def preprocess_text(text):
    """Предобработка текста"""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text


def text_to_sequence(text, vocab, max_len=100):
    """Конвертация текста в последовательность индексов"""
    words = preprocess_text(text).split()
    sequence = [vocab.get(word, vocab.get('<UNK>', 0)) for word in words]

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


def normalize_slang(text):
    """Нормализация интернет-сленга"""
    slang_map = {
        r'\btmr\b': 'tomorrow',
        r'\bcya\b': 'see you',
        r'\bye\b': 'yes',
        r'\byep\b': 'yes',
        r'\bu\b': 'you',
        r'\bur\b': 'your',
        r'\bpls\b': 'please',
        r'\bthx\b': 'thanks',
        r'\bok\b': 'okay',
        r'\bill\b': 'I will',
    }

    normalized = text
    for pattern, replacement in slang_map.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    return normalized


def rephrase_text(text, category='work', max_length=128):
    """Перефразирование текста с помощью HuggingFace T5 API"""

    try:
        # Нормализация сленга
        normalized = normalize_slang(text)

        # Промпт для T5
        prompt = f"paraphrase: {normalized}"

        # Запрос к API
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": max_length,
                "temperature": 0.7,
                "num_beams": 5
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
                paraphrased = result[0].get('generated_text', text)
                return paraphrased.strip()
            elif isinstance(result, dict) and 'generated_text' in result:
                return result['generated_text'].strip()
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
    """Генерация обращения"""
    if category == 'work':
        if recipient_name:
            return f"Dear {recipient_name},"
        return "Dear Sir/Madam,"
    elif category == 'personal':
        if recipient_name:
            return f"Hi {recipient_name},"
        return "Hi,"
    else:
        return ""


def generate_signature(sender_name=None, category='work'):
    """Генерация подписи"""
    if category == 'work':
        return f"\n\nBest regards,\n{sender_name or 'User'}"
    elif category == 'personal':
        return f"\n\nCheers,\n{sender_name or 'User'}"
    else:
        return ""


# === API ENDPOINTS ===

@app.route('/health', methods=['GET'])
def health():
    """Проверка работоспособности API"""
    return jsonify({'status': 'ok', 'message': 'Email Assistant API is running'})


@app.route('/classify', methods=['POST'])
def classify():
    """
    Классификация email

    Request:
    {
        "text": "Email text to classify"
    }

    Response:
    {
        "category": "work",
        "confidence": 92.5,
        "all_probabilities": {...}
    }
    """
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
    """
    Перефразирование текста

    Request:
    {
        "text": "Draft text",
        "category": "work" (optional, default: "work")
    }

    Response:
    {
        "original": "...",
        "rephrased": "...",
        "category": "work"
    }
    """
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
    """
    Полная композиция письма (классификация + перефразирование + оформление)

    Request:
    {
        "draft": "Short draft text",
        "recipient_name": "John Doe" (optional),
        "sender_name": "Your Name" (optional)
    }

    Response:
    {
        "draft": "...",
        "category": "work",
        "confidence": 92.5,
        "greeting": "Dear John Doe,",
        "body": "Rephrased text",
        "signature": "Best regards,\nYour Name",
        "full_email": "Complete formatted email"
    }
    """
    try:
        data = request.json
        draft = data.get('draft', '')
        recipient_name = data.get('recipient_name')
        sender_name = data.get('sender_name', 'User')

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
    """
    Обработка входящего email (классификация + применение метки)

    Request:
    {
        "subject": "Email subject",
        "body": "Email body"
    }

    Response:
    {
        "category": "work",
        "confidence": 92.5,
        "suggested_label": "WORK",
        "action": "Apply label and move to category folder"
    }
    """
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
    print("Email Assistant API Server")
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