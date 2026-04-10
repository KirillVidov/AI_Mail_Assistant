"""
Автоматическая обработка входящих писем
Запускается в фоне и проверяет новые письма каждые N минут
"""

import time
import pickle
import torch
import re
import os
import base64
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Импорт архитектуры модели
from architecture import EmailClassifierCNN_LSTM

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

# Загрузка модели
print("Загрузка модели...")
checkpoint = torch.load('best_model.pth', map_location=torch.device('cpu'))

with open('data/processed/transfer_vocabulary.pkl', 'rb') as f:
    vocab = pickle.load(f)

CATEGORIES = ['work', 'personal', 'spam', 'promo']

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

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def text_to_sequence(text, vocab, max_len=100):
    words = preprocess_text(text).split()
    sequence = [vocab.get(word, vocab.get('<UNK>', 0)) for word in words]
    
    if len(sequence) < max_len:
        sequence += [vocab.get('<PAD>', 0)] * (max_len - len(sequence))
    else:
        sequence = sequence[:max_len]
    
    return torch.tensor([sequence], dtype=torch.long)

def classify_email(text):
    with torch.no_grad():
        input_seq = text_to_sequence(text, vocab)
        output = model(input_seq)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        return {
            'category': CATEGORIES[predicted_class],
            'confidence': confidence
        }

def get_gmail_service():
    """Авторизация Gmail"""
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
        # Проверить существующие метки
        results = service.users().labels().list(userId='me').execute()
        labels = results.get('labels', [])
        
        for label in labels:
            if label['name'] == label_name:
                return label['id']
        
        # Создать новую метку
        label_object = {
            'name': label_name,
            'labelListVisibility': 'labelShow',
            'messageListVisibility': 'show'
        }
        created_label = service.users().labels().create(userId='me', body=label_object).execute()
        return created_label['id']
    
    except Exception as e:
        print(f"Error with label: {e}")
        return None

def process_unread_emails(service):
    """Обработать непрочитанные письма"""
    try:
        # Получить непрочитанные письма без меток AI/*
        query = 'is:unread -label:AI/WORK -label:AI/PERSONAL -label:AI/SPAM -label:AI/PROMO'
        results = service.users().messages().list(userId='me', q=query, maxResults=10).execute()
        messages = results.get('messages', [])
        
        if not messages:
            print("No new emails to process")
            return 0
        
        processed = 0
        for message in messages:
            msg_id = message['id']
            msg = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
            
            # Извлечь тему и тело
            headers = msg['payload']['headers']
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '')
            
            # Получить тело письма
            body = ""
            if 'parts' in msg['payload']:
                for part in msg['payload']['parts']:
                    if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                        break
            elif 'body' in msg['payload'] and 'data' in msg['payload']['body']:
                body = base64.urlsafe_b64decode(msg['payload']['body']['data']).decode('utf-8')
            
            # Классификация
            full_text = f"{subject} {body}"
            classification = classify_email(full_text)
            category = classification['category']
            confidence = classification['confidence']
            
            # Применить метку
            label_name = f"AI/{category.upper()}"
            label_id = get_or_create_label(service, label_name)
            
            if label_id:
                service.users().messages().modify(
                    userId='me',
                    id=msg_id,
                    body={'addLabelIds': [label_id]}
                ).execute()
                
                print(f"✓ Classified: '{subject[:50]}...' as {category} ({confidence*100:.1f}%)")
                processed += 1
            
        return processed
    
    except Exception as e:
        print(f"Error processing emails: {e}")
        return 0

def main():
    """Главный цикл"""
    print("="*60)
    print("Email Auto-Classifier Service")
    print("="*60)
    print("\nStarting automatic email classification...")
    print("Press Ctrl+C to stop\n")
    
    service = get_gmail_service()
    check_interval = 60  # Проверять каждые 60 секунд
    
    try:
        while True:
            print(f"\n[{time.strftime('%H:%M:%S')}] Checking for new emails...")
            processed = process_unread_emails(service)
            
            if processed > 0:
                print(f"✓ Processed {processed} email(s)")
            
            print(f"Next check in {check_interval} seconds...")
            time.sleep(check_interval)
    
    except KeyboardInterrupt:
        print("\n\nService stopped by user")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == '__main__':
    import os
    main()
