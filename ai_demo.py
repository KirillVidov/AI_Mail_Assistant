import torch
import pickle
import re


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


class EmailAssistant:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier = None
        self.classifier_vocab = None
        self.generator = None
        self.generator_vocab = None

    def load_classifier(self):
        print("Loading classifier...")

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

        print(f"Classifier loaded (accuracy: {checkpoint['val_acc']:.1f}%)")

    def load_generator(self):
        print("Loading generator...")

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

        print(f"Generator loaded (train loss: {checkpoint['train_loss']:.2f})")

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

    def process_email(self, subject, body=""):
        print("\n" + "=" * 60)
        print("EMAIL ASSISTANT")
        print("=" * 60)

        print(f"\nIncoming Email:")
        print(f"Subject: {subject}")
        if body:
            print(f"Body: {body[:200]}{'...' if len(body) > 200 else ''}")

        category, confidence = self.classify_email(subject, body)

        print(f"\nClassification:")
        print(f"Category: {category.upper()}")
        print(f"Confidence: {confidence:.1%}")

        if category != 'spam':
            full_text = f"{subject} {body}"
            reply = self.generate_reply(full_text)

            print(f"\nGenerated Reply:")
            print(f"{reply}")
        else:
            print(f"\nAction: Email marked as SPAM - no reply generated")

        print("=" * 60)

        return category, reply if category != 'spam' else None


def demo():
    assistant = EmailAssistant()

    assistant.load_classifier()
    assistant.load_generator()

    print("\n\nSystem ready!")
    print("\nTry some examples:\n")

    examples = [
        {
            'subject': 'Meeting tomorrow at 2pm',
            'body': 'Hi, can we reschedule our meeting? I have a conflict at 2pm tomorrow.'
        },
        {
            'subject': '50% OFF SALE - LIMITED TIME!!!',
            'body': 'Click here now to get amazing discounts! Free shipping on all orders!'
        },
        {
            'subject': 'Lunch this weekend?',
            'body': 'Hey! Want to grab lunch on Saturday? Let me know what works for you.'
        },
        {
            'subject': 'Q4 Report Review',
            'body': 'Please review the attached Q4 financial report and send feedback by Friday.'
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n\n{'*' * 60}")
        print(f"EXAMPLE {i}")
        print('*' * 60)

        assistant.process_email(example['subject'], example['body'])

    print("\n\nInteractive mode:")
    print("Enter your own emails (or 'quit' to exit)\n")

    while True:
        subject = input("\nEmail subject: ").strip()
        if subject.lower() == 'quit':
            break

        body = input("Email body (optional): ").strip()

        assistant.process_email(subject, body)


if __name__ == "__main__":
    demo()