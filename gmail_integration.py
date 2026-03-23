import os
import pickle
import base64
import torch
import re
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email.mime.text import MIMEText

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

class ReplyVocabulary:
    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}

    def encode(self, text, max_length, add_sos_eos=False):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()

        indices = [self.word2idx.get(token, 1) for token in tokens]

        if add_sos_eos:
            indices = [2] + indices + [3]

        if len(indices) < max_length:
            indices += [0] * (max_length - len(indices))
        else:
            indices = indices[:max_length]

        return indices

    def decode(self, indices):
        words = []
        for idx in indices:
            if idx == 3:
                break
            if idx in [0, 2]:
                continue
            words.append(self.idx2word.get(idx, '<UNK>'))
        return ' '.join(words)


class GmailEmailAssistant:
    def __init__(self):
        self.gmail_service = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier = None
        self.classifier_vocab = None
        self.generator = None
        self.generator_vocab = None

    def authenticate_gmail(self):
        print("Authenticating with Gmail...")

        creds = None
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists('credentials.json'):
                    print("\nError: credentials.json not found!")
                    print("Please follow setup instructions to get credentials.json")
                    return False

                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                creds = flow.run_local_server(port=8080)

            with open('token.json', 'w') as token:
                token.write(creds.to_json())

        self.gmail_service = build('gmail', 'v1', credentials=creds)
        print("Gmail authenticated!")
        return True

    def load_models(self):
        print("Loading AI models...")

        checkpoint = torch.load('best_model.pth', map_location=self.device)

        with open('data/processed/transfer_vocabulary.pkl', 'rb') as f:
            vocab_data = pickle.load(f)
            self.classifier_vocab = vocab_data['word2idx']

        from architecture import EmailClassifierCNN_LSTM

        self.classifier = EmailClassifierCNN_LSTM(
            vocab_size=len(self.classifier_vocab),
            embedding_dim=128,
            hidden_dim=128,
            num_classes=4,
            dropout=0.5
        ).to(self.device)

        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.eval()

        checkpoint = torch.load('best_seq2seq_model.pth', map_location=self.device)

        with open('reply_vocabulary.pkl', 'rb') as f:
            self.generator_vocab = pickle.load(f)

        from seq2seq_architecture import create_seq2seq_model

        self.generator = create_seq2seq_model(
            vocab_size=len(self.generator_vocab.word2idx),
            embedding_dim=256,
            hidden_dim=512,
            num_layers=2,
            dropout=0.3,
            device=self.device
        )

        self.generator.load_state_dict(checkpoint['model_state_dict'])
        self.generator.eval()

        print("Models loaded!")

    def get_unread_emails(self, max_results=10):
        results = self.gmail_service.users().messages().list(
            userId='me',
            labelIds=['INBOX'],
            q='is:unread',
            maxResults=max_results
        ).execute()

        messages = results.get('messages', [])
        return messages

    def get_email_content(self, msg_id):
        message = self.gmail_service.users().messages().get(
            userId='me',
            id=msg_id,
            format='full'
        ).execute()

        headers = message['payload']['headers']
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '')
        sender = next((h['value'] for h in headers if h['name'] == 'From'), '')

        body = ''
        if 'parts' in message['payload']:
            for part in message['payload']['parts']:
                if part['mimeType'] == 'text/plain':
                    body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                    break
        else:
            if 'body' in message['payload'] and 'data' in message['payload']['body']:
                body = base64.urlsafe_b64decode(message['payload']['body']['data']).decode('utf-8')

        return {
            'id': msg_id,
            'subject': subject,
            'from': sender,
            'body': body
        }

    def classify_email(self, subject, body=""):
        text = f"{subject} {body}".lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()

        indices = [self.classifier_vocab.get(token, 1) for token in tokens]

        if len(indices) < 128:
            indices += [0] * (128 - len(indices))
        else:
            indices = indices[:128]

        input_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)

        with torch.no_grad():
            output = self.classifier(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)

        categories = {0: 'work', 1: 'personal', 2: 'spam', 3: 'promo'}
        category = categories[prediction.item()]
        conf = confidence.item()

        return category, conf

    def generate_reply(self, email_text):
        src_encoded = self.generator_vocab.encode(email_text, max_length=100, add_sos_eos=False)
        src_tensor = torch.tensor([src_encoded], dtype=torch.long).to(self.device)

        generated_indices = self.generator.generate(
            src_tensor,
            max_length=50,
            sos_token=2,
            eos_token=3,
            temperature=0.8,
            repetition_penalty=1.5
        )

        reply = self.generator_vocab.decode(generated_indices)
        return reply

    def apply_label(self, msg_id, category):
        label_map = {
            'work': 'Work',
            'personal': 'Personal',
            'spam': 'Spam',
            'promo': 'Promotions'
        }

        label_name = label_map.get(category, 'Unknown')

        labels = self.gmail_service.users().labels().list(userId='me').execute()
        label_id = None

        for label in labels.get('labels', []):
            if label['name'] == label_name:
                label_id = label['id']
                break

        if not label_id:
            label = self.gmail_service.users().labels().create(
                userId='me',
                body={'name': label_name}
            ).execute()
            label_id = label['id']

        self.gmail_service.users().messages().modify(
            userId='me',
            id=msg_id,
            body={'addLabelIds': [label_id]}
        ).execute()

    def process_inbox(self):
        print("\nProcessing inbox...")

        messages = self.get_unread_emails(max_results=10)

        if not messages:
            print("No unread emails")
            return

        print(f"Found {len(messages)} unread emails")

        for i, msg in enumerate(messages, 1):
            print(f"\n[{i}/{len(messages)}] Processing email...")

            email = self.get_email_content(msg['id'])

            print(f"From: {email['from']}")
            print(f"Subject: {email['subject'][:50]}...")

            category, confidence = self.classify_email(email['subject'], email['body'])

            print(f"Category: {category.upper()} (confidence: {confidence:.1%})")

            self.apply_label(msg['id'], category)
            print(f"Label applied: {category}")

            if category not in ['spam'] and confidence > 0.7:
                reply = self.generate_reply(f"{email['subject']} {email['body']}")
                print(f"Generated reply: {reply[:80]}...")
            else:
                print("No reply generated (spam or low confidence)")

        print("\nProcessing complete!")


def setup_instructions():
    print("=" * 60)
    print("GMAIL INTEGRATION SETUP")
    print("=" * 60)
    print("\nTo use this script, you need to:")
    print("\n1. Go to: https://console.cloud.google.com")
    print("2. Create a new project (or select existing)")
    print("3. Enable Gmail API:")
    print("   - APIs & Services > Enable APIs and Services")
    print("   - Search 'Gmail API' > Enable")
    print("\n4. Create credentials:")
    print("   - APIs & Services > Credentials")
    print("   - Create Credentials > OAuth client ID")
    print("   - Application type: Desktop app")
    print("   - Download JSON")
    print("\n5. Rename downloaded file to 'credentials.json'")
    print("6. Place it in this project folder")
    print("\n7. Install dependencies:")
    print("   pip install google-auth-oauthlib google-auth google-api-python-client")
    print("\n8. Run this script again")
    print("=" * 60)


def main():
    if not os.path.exists('credentials.json'):
        setup_instructions()
        return

    assistant = GmailEmailAssistant()

    if not assistant.authenticate_gmail():
        return

    assistant.load_models()

    print("\nGmail Email Assistant is ready!")
    print("\nOptions:")
    print("1. Process inbox now")
    print("2. Run continuously (check every 5 minutes)")

    choice = input("\nChoice (1 or 2): ").strip()

    if choice == '1':
        assistant.process_inbox()
    elif choice == '2':
        import time
        print("\nRunning continuously (Ctrl+C to stop)...")
        while True:
            try:
                assistant.process_inbox()
                print("\nWaiting 5 minutes...")
                time.sleep(300)
            except KeyboardInterrupt:
                print("\nStopped")
                break


if __name__ == "__main__":
    main()