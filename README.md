#  Email Assistant — Интеллектуальная система классификации и обработки электронной почты

Система автоматически классифицирует входящие письма Gmail по 4 категориям (work / personal / promo / spam) и помогает составлять ответы через YandexGPT API.

---

##  Структура проекта

```
AI_Mail_Assistant/
├── flask_api_russian.py              # REST API сервер (Flask)
├── auto_classifier_service_russian.py # Фоновый сервис классификации
├── Code.gs                           # Gmail Add-on (Google Apps Script)
├── appsscript.json                   # Манифест Add-on
├── best_model.pth                    # Веса обученной модели (7.2 МБ)
├── vocabulary.pkl                    # Словарь для токенизации (156 КБ)
├── credentials.json                  # OAuth credentials Gmail API (не в репо)
├── token.json                        # OAuth токен (генерируется автоматически)
├── .env                              # API ключ YandexGPT (не в репо)
├── .gitignore
├── requirements.txt
└── README.md
```

---

##  Установка

### 1. Клонировать репозиторий

```bash
git clone https://github.com/KirillVidov/AI_Mail_Assistant
cd AI_Mail_Assistant
```

### 2. Установить зависимости

```bash
pip install -r requirements.txt
```

### 3. Настроить Gmail API

1. Перейти на [Google Cloud Console](https://console.cloud.google.com/)
2. Создать проект **Email Assistant**
3. Включить **Gmail API**
4. Создать OAuth 2.0 credentials (тип: Desktop app)
5. Скачать `credentials.json` и поместить в корень проекта

### 4. Настроить YandexGPT API

Создать файл `.env` в корне проекта:

```
YANDEX_API_KEY=ваш_api_ключ
```

Получить ключ можно в [Яндекс Cloud](https://cloud.yandex.ru/).

### 5. Добавить файлы модели

Поместить в корень проекта:
- `best_model.pth` — веса модели
- `vocabulary.pkl` — словарь токенизатора

---

##  Запуск

Открыть три отдельных терминала:

**Терминал 1 — Flask API:**
```bash
python flask_api_russian.py
# Сервер запустится на http://localhost:5000
```

**Терминал 2 — Auto-Classifier:**
```bash
python auto_classifier_service_russian.py
# При первом запуске откроется браузер для авторизации Gmail
```

**Терминал 3 — cloudflare туннель:**
```bash
cloudflared tunnel --protocol http2 --url http://localhost:5000
ngrok http 5000
# Скопировать URL вида https://abc123.ngrok-free.app
```

---

##  Установка Gmail Add-on

1. Перейти на [Google Apps Script](https://script.google.com/)
2. Создать новый проект
3. Скопировать содержимое `Code.gs`
4. Обновить константу `API_URL` на актуальный URL ngrok:
   ```javascript
   const API_URL = 'https://abc123.ngrok-free.app';
   ```
5. Скопировать содержимое `appsscript.json` в манифест проекта
6. Нажать **Развернуть → Тестовое развёртывание → Установить**

После установки при открытии любого письма в Gmail справа появится боковая панель **Email Assistant**.

---

## 📡 REST API

| Метод | Эндпоинт | Описание |
|-------|----------|----------|
| GET | `/health` | Проверка работоспособности |
| POST | `/classify` | Классификация текста письма |
| POST | `/rephrase` | Перефразирование черновика |
| POST | `/compose` | Генерация полного письма |
| POST | `/process_email` | Обработка входящего письма |

### Пример запроса к /classify

```bash
curl -X POST http://localhost:5000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Please send me the report by Friday"}'
```

Ответ:
```json
{
  "category": "work",
  "confidence": 94.2,
  "all_probabilities": {
    "work": 94.2,
    "personal": 3.1,
    "spam": 1.8,
    "promo": 0.9
  }
}
```

### Пример запроса к /compose

```bash
curl -X POST http://localhost:5000/compose \
  -H "Content-Type: application/json" \
  -d '{
    "draft": "ок, сделаю завтра",
    "sender_name": "Иван",
    "recipient_name": "Пётр Иванович",
    "force_category": "work"
  }'
```

---

##  Модель классификации

- **Архитектура:** CNN + BiLSTM + Attention
- **Параметры:** 1 847 204
- **Датасет:** Enron Email Dataset (12 000 писем, 4 категории)
- **Точность:** 92,3% на валидационной выборке
- **Обучение:** Google Colab, GPU Tesla T4, 4 мин 37 сек

### Категории

| Категория | Описание | Ярлык Gmail |
|-----------|----------|-------------|
| work | Деловая переписка | AI/WORK |
| personal | Личные письма | AI/PERSONAL |
| promo | Рекламные рассылки | AI/PROMO |
| spam | Нежелательная почта | AI/SPAM |

---

##  Технические требования

| Компонент | Минимум |
|-----------|---------|
| Python | 3.11+ |
| RAM | 4 ГБ |
| Диск | 2 ГБ свободно |
| ОС | Windows 10/11 или Ubuntu 20.04+ |
| Браузер | Google Chrome 90+ |
| Интернет | 5 Мбит/с+ |

---

##  Диагностика

| Проблема | Решение |
|----------|---------|
| Flask API не запускается | Проверить наличие `best_model.pth` и `vocabulary.pkl` |
| Auto-Classifier не применяет метки | Проверить `token.json` и авторизацию Gmail |
| Add-on не отвечает | Проверить запущен ли ngrok, обновить `API_URL` в `Code.gs` |
| YandexGPT не работает | Проверить `.env` и корректность API-ключа |

---

##  Лицензия

Проект разработан в рамках выпускной квалификационной работы.  
ЧОУ ВО «Московский университет им. С.Ю. Витте», 2026.
