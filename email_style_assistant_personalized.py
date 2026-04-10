import torch
import pickle
import re
import os
import pandas as pd
from transformers import pipeline


class PersonalizedEmailAssistant:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier = None
        self.classifier_vocab = None
        self.rephraser = None
        self.user_examples = {}

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

    def load_rephraser(self):
        print("Loading paraphrase model...")
        print("This may take a few minutes on first run...")

        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        model_name = "humarin/chatgpt_paraphraser_on_T5_base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if torch.cuda.is_available():
            self.t5_model = self.t5_model.to('cuda')

        print("Paraphrase model loaded!")

    def load_user_style(self):
        """
        Загружает примеры писем пользователя для персонализации
        """
        user_file = 'data/processed/user_sent_emails.csv'

        if not os.path.exists(user_file):
            print("\nNo user emails found - using standard style")
            return

        print("\nLoading user's writing style...")

        try:
            df = pd.read_csv(user_file)

            for category in ['work', 'personal', 'promo']:
                category_emails = df[df['category'] == category]

                if len(category_emails) > 0:
                    examples = []
                    for _, row in category_emails.head(3).iterrows():
                        if len(row['body']) > 20 and len(row['body']) < 300:
                            examples.append(row['body'])

                    if examples:
                        self.user_examples[category] = examples
                        print(f"  {category}: {len(examples)} examples loaded")

            if self.user_examples:
                print(f"\nPersonalization enabled for: {', '.join(self.user_examples.keys())}")
            else:
                print("\nNo suitable examples found - using standard style")

        except Exception as e:
            print(f"Error loading user emails: {e}")

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

    def build_personalized_prompt(self, draft, category):
        """
        Создаёт промпт для перефразирования
        """
        if category in self.user_examples and len(self.user_examples[category]) > 0:
            examples = self.user_examples[category]

            prompt = f"""paraphrase: {draft}"""

            return prompt, True
        else:
            prompt = f"paraphrase: {draft}"

            return prompt, False

    def rephrase_text(self, draft, category):
        if category == 'spam':
            return "[SPAM detected - no rephrasing needed]", False

        prompt, is_personalized = self.build_personalized_prompt(draft, category)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}

            outputs = self.t5_model.generate(
                **inputs,
                max_length=150,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )

            rephrased = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return rephrased, is_personalized
        except Exception as e:
            print(f"Rephrasing error: {e}")
            return draft, False

    def process_draft(self, subject, body):
        print("\n" + "=" * 60)
        print("PERSONALIZED EMAIL ASSISTANT")
        print("=" * 60)

        print(f"\nOriginal Draft:")
        print(f"Subject: {subject}")
        print(f"Body: {body}")

        category, confidence = self.classify_email(subject, body)

        print(f"\nClassification:")
        print(f"Category: {category.upper()}")
        print(f"Confidence: {confidence:.1%}")

        if category != 'spam':
            print(f"\nRephrasing in {category} style...")

            rephrased, is_personalized = self.rephrase_text(body, category)

            print(f"\nRephrased Email:")
            if is_personalized:
                print(f"[Using your personal writing style]")
            else:
                print(f"[Using standard {category} style]")
            print(f"Subject: {subject}")
            print(f"Body: {rephrased}")
        else:
            print(f"\nAction: Marked as SPAM - no rephrasing needed")
            rephrased = None

        print("=" * 60)

        return {
            'category': category,
            'confidence': confidence,
            'original': body,
            'rephrased': rephrased
        }


def demo():
    print("\n" + "=" * 60)
    print("PERSONALIZED EMAIL STYLE ASSISTANT")
    print("Free Version with User Style Adaptation")
    print("=" * 60)

    assistant = PersonalizedEmailAssistant()

    assistant.load_classifier()
    assistant.load_rephraser()
    assistant.load_user_style()

    print("\n\nDEMO: Testing with example drafts\n")

    test_cases = [
        {
            'subject': 'Meeting',
            'body': 'hey can we meet tmrw at 2? need to discuss project'
        },
        {
            'subject': 'Report',
            'body': 'pls send me the Q4 report when u get a chance thx'
        },
        {
            'subject': 'Weekend Plans',
            'body': 'wanna hang out this weekend? maybe grab lunch'
        },
        {
            'subject': 'Update',
            'body': 'just wanted to let u know the task is done'
        }
    ]

    for i, case in enumerate(test_cases, 1):
        print(f"\n{'*' * 60}")
        print(f"EXAMPLE {i}")
        print('*' * 60)

        result = assistant.process_draft(case['subject'], case['body'])

    print("\n\nINTERACTIVE MODE")
    print("Enter your email drafts (or 'quit' to exit)\n")

    while True:
        subject = input("\nSubject: ").strip()
        if subject.lower() == 'quit':
            break

        body = input("Body: ").strip()

        if subject or body:
            assistant.process_draft(subject, body)


if __name__ == "__main__":
    demo()