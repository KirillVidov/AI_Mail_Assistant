# Gmail Add-on для Email Assistant - Инструкция по установке

## 🎯 Что это дает?

После установки в Gmail появится боковая панель с функциями:
- ✅ Автоматическая классификация писем (work/personal/spam/promo)
- ✅ Применение меток одним кликом
- ✅ Генерация ответов в правильном стиле
- ✅ Чат-интерфейс: пишешь кратко → получаешь оформленное письмо

---

## 📋 Установка - Шаг за шагом

### Шаг 1: Запустить Python API

```bash
# 1. Установить зависимости
pip install flask flask-cors torch transformers

# 2. Убедиться что есть файлы:
#    - best_model.pth
#    - transfer_vocabulary.pkl
#    - (T5 модель загрузится автоматически)

# 3. Запустить API
python flask_api.py

# Должно появиться:
# Server running on http://localhost:5000
```

**Оставь терминал открытым!** API должен работать постоянно.

---

### Шаг 2: Создать Gmail Add-on

#### 2.1 Открыть Google Apps Script

1. Перейди на https://script.google.com
2. Нажми **"Новый проект"**
3. Название проекта: **"Email Assistant"**

#### 2.2 Добавить код

1. **Удали** весь код в `Code.gs`
2. **Скопируй** весь код из файла `gmail_addon.gs`
3. **Вставь** в редактор

#### 2.3 Добавить конфигурацию

1. Слева в меню → нажми **⚙️ (Settings)**
2. Поставь галочку **"Show appsscript.json manifest file"**
3. Слева появится файл `appsscript.json` → открой его
4. **Удали** весь код
5. **Скопируй** весь код из файла `appsscript.json`
6. **Вставь** в редактор

#### 2.4 Настроить API URL

В файле `Code.gs` найди строку:
```javascript
const API_URL = 'http://localhost:5000';
```

**Для локального тестирования** оставь как есть.

**Для продакшена** (если развернул API в облаке):
```javascript
const API_URL = 'https://your-api-url.com';
```

---

### Шаг 3: Тестовое развертывание

1. В Google Apps Script нажми **Deploy** → **Test deployments**
2. Нажми **Install**
3. Выбери свой Google аккаунт
4. Разреши доступ к Gmail (нажми **Allow**)

---

### Шаг 4: Использование

#### Открой Gmail:
1. Открой любое письмо
2. Справа появится панель **Email Assistant** 📧
3. Увидишь:
   - **Category**: work/personal/spam/promo
   - **Confidence**: процент уверенности
   - Кнопка **Apply Label** - применить метку

#### Создать ответ:
1. В поле **"Your draft"** напиши кратко: `ok ill come tomorrow`
2. Нажми **Generate Reply**
3. Получишь оформленное письмо:
   ```
   Dear [Recipient],
   
   Certainly, I will come tomorrow.
   
   Best regards,
   [Your Name]
   ```

---

## 🔧 Альтернатива: Развернуть API в облаке

Если не хочешь держать Python запущенным постоянно, разверни API бесплатно:

### Вариант A: Railway.app (Рекомендую)

```bash
# 1. Создай requirements.txt
flask
flask-cors
torch
transformers

# 2. Создай Procfile
web: python flask_api.py

# 3. Зарегистрируйся на railway.app
# 4. Deploy from GitHub
# 5. Получишь URL типа: https://email-assistant.railway.app
```

### Вариант B: Render.com

То же самое, бесплатный тариф.

### Вариант C: PythonAnywhere

Бесплатный план, но медленнее.

---

## 🐛 Решение проблем

### Problem: "API not responding"
**Solution**: Проверь что `flask_api.py` запущен и работает на http://localhost:5000

### Problem: "CORS error"
**Solution**: Убедись что в `flask_api.py` есть строка `CORS(app)`

### Problem: "Label already exists"
**Solution**: Это нормально, метка создается только один раз

### Problem: "Can't access from Google Apps Script"
**Solution**: Если API на localhost, Google Apps Script не сможет к нему обратиться.
Нужно либо:
1. Использовать ngrok для туннеля: `ngrok http 5000`
2. Развернуть API в облаке (Railway/Render)

---

## 📊 Для диплома

В дипломе можешь написать:

> **Интеграция с Gmail**
> 
> Система интегрирована с Gmail через Gmail Add-on на базе Google Apps Script.
> Архитектура решения:
> 
> 1. **Frontend**: Gmail Add-on (JavaScript/Google Apps Script)
> 2. **Backend**: Flask REST API (Python)
> 3. **Модели**: PyTorch классификатор + HuggingFace T5
> 
> **Функциональные возможности:**
> - Автоматическая классификация входящих писем
> - Применение меток одним кликом
> - Генерация ответов в чат-стиле интерфейса
> - Персонализация на основе истории переписки
> 
> **Развертывание:**
> - Локальное тестирование: localhost
> - Продакшен: Railway.app / Render.com (бесплатно)
> - OAuth 2.0 для безопасного доступа к Gmail

---

## 🚀 Продвинутые возможности

Можно добавить позже:
- ⏰ Автоматическая обработка каждые N минут
- 📧 Автоответы на определенные типы писем
- 📊 Статистика по категориям
- 🎨 Разные шаблоны для разных получателей
- 🔔 Уведомления о важных письмах

---

## ✅ Готово!

Теперь у тебя полноценная интеграция с Gmail без ручного запуска Python!
