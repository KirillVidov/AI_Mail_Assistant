"""
Генератор синтетического датасета русских email
Создает разнообразные примеры писем для обучения модели классификации
"""

import csv
import random
from datetime import datetime

# Шаблоны для каждой категории

WORK_TEMPLATES = [
    # Встречи и планирование
    "Добрый день! Перенесём встречу на {day} в {time}?",
    "Здравствуйте, нужно обсудить {project}. Когда вам удобно?",
    "Уважаемые коллеги, встреча по {topic} состоится {day} в {time} в {room}",
    "Прошу подтвердить участие в совещании {day}",

    # Отчеты и документы
    "Отправляю отчет по {project}. Прошу ознакомиться до {deadline}",
    "Необходимо подготовить презентацию к {day}",
    "Прикрепляю документы по {topic}. Жду ваших комментариев",
    "Пожалуйста, проверьте бюджет {project} до конца недели",

    # Задачи
    "Нужно завершить {task} до {deadline}",
    "Прошу назначить ответственного за {project}",
    "Задача {task} требует вашего внимания",
    "Напоминаю о дедлайне по {project} - {day}",

    # Согласование
    "Прошу согласовать {document}",
    "Требуется ваша подпись на {document}",
    "Согласуйте, пожалуйста, смету по {project}",

    # Вопросы
    "Есть вопрос по {topic}. Можем обсудить?",
    "Как продвигается работа над {project}?",
    "Нужна помощь с {task}",
    "Уточните, пожалуйста, сроки по {project}",

    # Результаты
    "Проект {project} завершен. Результаты во вложении",
    "Отправляю итоги совещания от {day}",
    "Статистика за {month} в приложении",
]

PERSONAL_TEMPLATES = [
    # Приветствия
    "Привет! Как дела? Давно не виделись!",
    "Здравствуй! Как настроение?",
    "Хай! Что нового?",
    "Приветик! Как выходные прошли?",

    # Планы
    "Пойдем {activity} в {day}?",
    "Может встретимся {day}?",
    "Планы на выходные есть?",
    "Хочу пригласить тебя {activity}",

    # Благодарности
    "Спасибо большое за {gift}!",
    "Благодарю за помощь с {task}!",
    "Очень признателен за поддержку",
    "Спс, выручил!",

    # Вопросы
    "Как там {person}?",
    "Ты {activity} сегодня?",
    "Подскажи, где можно {action}?",

    # События
    "Поздравляю с {event}!",
    "С днем рождения! Желаю {wish}!",
    "Классно провели время вчера!",

    # Просьбы
    "Не мог бы ты помочь с {task}?",
    "Можешь одолжить {item}?",
    "Подскажи, как {action}?",
]

SPAM_TEMPLATES = [
    # Выигрыши
    "ПОЗДРАВЛЯЕМ! Вы выиграли {amount} рублей! Перейдите по ссылке",
    "Вы победитель лотереи! Приз {amount}₽",
    "Срочно! Ваш выигрыш {amount} ждет вас!",

    # Фишинг
    "Ваш аккаунт заблокирован. Подтвердите данные по ссылке",
    "ВНИМАНИЕ! Подозрительная активность. Пройдите верификацию",
    "Ваша карта будет заблокирована. Обновите данные",

    # Нигерийские письма
    "Здравствуйте! Я принц из {country}. Нужна ваша помощь перевести {amount}$",
    "Получите наследство {amount} евро. Оплатите пошлину {fee}₽",

    # Займы
    "Деньги в долг без проверок! До {amount}₽ за 5 минут!",
    "Кредит {amount}₽ одобрен! Получите сегодня",
    "Займ на карту моментально! {amount}₽",

    # Лекарства
    "Виагра {price}₽! Доставка анонимно!",
    "Таблетки для похудения! Минус {weight}кг за неделю!",

    # Подозрительное
    "Проверьте вложение СРОЧНО",
    "Важная информация в прикрепленном файле",
    "Откройте документ немедленно",
]

PROMO_TEMPLATES = [
    # Скидки
    "Скидка {discount}% на {product}!",
    "Только сегодня! {product} со скидкой {discount}%",
    "Распродажа! Все товары -{discount}%!",
    "Финальная распродажа! До {discount}% на всё!",

    # Акции
    "Акция! При покупке {product} - {gift} в подарок!",
    "2 по цене 1 на {product}!",
    "Купи {product1} и получи {product2} бесплатно!",

    # Новинки
    "Новинка! {product} уже в продаже!",
    "Встречайте новый {product}!",
    "Эксклюзивно у нас: {product}!",

    # Предложения
    "Специальное предложение для вас: {product} всего {price}₽!",
    "Персональная скидка {discount}% на {product}",
    "Только для наших клиентов: {product} по супер-цене!",

    # Ограниченные
    "Осталось {count} {product} по акции!",
    "Предложение действует до {date}!",
    "Последний день скидок! Успейте купить {product}!",

    # Подписки
    "Подпишитесь на рассылку и получите {discount}% скидку!",
    "Бонус {amount}₽ за регистрацию!",
]

# Списки для заполнения шаблонов
PROJECTS = ['проект Alpha', 'внедрение CRM', 'разработка сайта', 'маркетинговая кампания', 'запуск продукта',
            'аудит', 'миграция данных', 'оптимизация процессов', 'обучение персонала', 'ребрендинг']

TOPICS = ['бюджет', 'стратегия', 'новый функционал', 'планирование', 'закупки', 'HR-вопросы',
          'техническая документация', 'клиентский сервис', 'аналитика', 'безопасность']

TASKS = ['подготовка отчета', 'согласование договора', 'проверка данных', 'обновление документации',
         'тестирование', 'код-ревью', 'анализ метрик', 'подготовка презентации', 'исправление ошибок']

DOCUMENTS = ['договор', 'смета', 'техническое задание', 'план проекта', 'отчет', 'презентация',
             'бюджет', 'спецификация', 'инструкция']

ROOMS = ['переговорная 1', 'конференц-зал', 'переговорная на 3 этаже', 'зал А', 'офис CEO', 'Zoom']

DAYS = ['завтра', 'в понедельник', 'во вторник', 'в среду', 'в четверг', 'в пятницу',
        'на следующей неделе', 'послезавтра', '15 числа', 'в конце месяца']

TIMES = ['10:00', '11:00', '14:00', '15:00', '16:00', '12:30', '15:30']

MONTHS = ['январь', 'февраль', 'март', 'апрель', 'май', 'июнь',
          'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь']

ACTIVITIES = ['в кино', 'поужинать', 'на концерт', 'погулять', 'в кафе', 'на выставку',
              'в театр', 'на каток', 'в парк', 'на пикник']

GIFTS = ['подарок', 'книгу', 'помощь', 'совет', 'поддержку', 'цветы']

EVENTS = ['днем рождения', 'Новым годом', 'повышением', '8 марта', 'свадьбой', 'Днем защитника отечества']

WISHES = ['счастья', 'здоровья', 'успехов', 'удачи', 'всего наилучшего', 'исполнения желаний']

PERSONS = ['мама', 'папа', 'сестра', 'брат', 'друг', 'Маша', 'Саша', 'родители']

ACTIONS = ['добраться до', 'купить', 'найти', 'сделать', 'починить', 'установить']

ITEMS = ['книгу', 'ноутбук', 'инструмент', 'машину', 'телефон']

COUNTRIES = ['Нигерия', 'Кения', 'Замбия', 'Уганда']

PRODUCTS = ['iPhone', 'ноутбук', 'кроссовки', 'куртка', 'духи', 'часы', 'телевизор', 'наушники',
            'пылесос', 'кофемашина', 'игровая консоль', 'планшет', 'смартфон']


def generate_email(category, template):
    """Генерация одного email из шаблона"""
    text = template

    # Замена плейсхолдеров
    replacements = {
        '{project}': random.choice(PROJECTS),
        '{topic}': random.choice(TOPICS),
        '{task}': random.choice(TASKS),
        '{document}': random.choice(DOCUMENTS),
        '{room}': random.choice(ROOMS),
        '{day}': random.choice(DAYS),
        '{time}': random.choice(TIMES),
        '{deadline}': random.choice(DAYS),
        '{month}': random.choice(MONTHS),
        '{activity}': random.choice(ACTIVITIES),
        '{gift}': random.choice(GIFTS),
        '{person}': random.choice(PERSONS),
        '{event}': random.choice(EVENTS),
        '{wish}': random.choice(WISHES),
        '{action}': random.choice(ACTIONS),
        '{item}': random.choice(ITEMS),
        '{amount}': str(random.choice([50000, 100000, 500000, 1000000, 5000000])),
        '{country}': random.choice(COUNTRIES),
        '{fee}': str(random.choice([5000, 10000, 15000])),
        '{discount}': str(random.choice([10, 20, 30, 40, 50, 70])),
        '{product}': random.choice(PRODUCTS),
        '{product1}': random.choice(PRODUCTS),
        '{product2}': random.choice(PRODUCTS),
        '{price}': str(random.choice([990, 1990, 2990, 4990, 9990])),
        '{count}': str(random.choice([3, 5, 10, 15])),
        '{date}': random.choice(['конца недели', 'воскресенья', '31 числа']),
        '{weight}': str(random.choice([5, 10, 15, 20])),
    }

    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)

    return text


def generate_dataset(samples_per_category=3000):
    """Генерация полного датасета"""
    print("🚀 Начинаем генерацию синтетического датасета русских email...")
    print(f"📊 Будет создано: {samples_per_category * 4} писем")

    dataset = []

    categories = {
        'work': WORK_TEMPLATES,
        'personal': PERSONAL_TEMPLATES,
        'spam': SPAM_TEMPLATES,
        'promo': PROMO_TEMPLATES
    }

    for category, templates in categories.items():
        print(f"\n✍️  Генерация категории: {category.upper()}")

        for i in range(samples_per_category):
            template = random.choice(templates)
            text = generate_email(category, template)

            dataset.append({
                'id': f"{category}_{i + 1}",
                'text': text,
                'category': category,
                'category_id': list(categories.keys()).index(category)
            })

            if (i + 1) % 500 == 0:
                print(f"  ✓ Создано {i + 1}/{samples_per_category}")

    # Перемешиваем
    random.shuffle(dataset)

    return dataset


def save_to_csv(dataset, filename='russian_emails_synthetic.csv'):
    """Сохранение в CSV"""
    print(f"\n💾 Сохранение в {filename}...")

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'text', 'category', 'category_id'])
        writer.writeheader()
        writer.writerows(dataset)

    print(f"✅ Датасет сохранен! Всего записей: {len(dataset)}")
    print(f"\n📈 Распределение по категориям:")

    from collections import Counter
    category_counts = Counter(row['category'] for row in dataset)
    for category, count in category_counts.items():
        percentage = (count / len(dataset)) * 100
        print(f"  {category.upper()}: {count} ({percentage:.1f}%)")


if __name__ == '__main__':
    # Генерация датасета
    dataset = generate_dataset(samples_per_category=3000)

    # Сохранение
    save_to_csv(dataset, 'data/processed/russian_emails_synthetic.csv')

