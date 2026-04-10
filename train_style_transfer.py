import torch
import torch.nn as nn
import pandas as pd
import pickle
import re
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


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


class StyleTransferDataset(Dataset):
    def __init__(self, df, vocabulary, max_length=100):
        self.sources = df['original_body'].tolist()
        self.targets = df['reply_body'].tolist()
        self.categories = df['category'].tolist() if 'category' in df.columns else ['work'] * len(df)
        self.vocabulary = vocabulary
        self.max_length = max_length

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        src = self.vocabulary.encode(self.sources[idx], self.max_length, add_sos_eos=False)
        trg = self.vocabulary.encode(self.targets[idx], self.max_length, add_sos_eos=True)

        return {
            'src': torch.tensor(src, dtype=torch.long),
            'trg': torch.tensor(trg, dtype=torch.long)
        }


def prepare_style_transfer_data():
    print("Preparing data for style transfer...")

    # Load email reply pairs (from Enron)
    df = pd.read_csv('data/processed/email_reply_pairs.csv')
    print(f"Loaded {len(df)} email pairs")

    # Add work category (Enron is business emails)
    df['category'] = 'work'

    # Load user emails if available
    try:
        user_df = pd.read_csv('data/processed/user_sent_emails.csv')
        print(f"Loaded {len(user_df)} user emails")

        # Create synthetic pairs from user emails
        # (same email as both source and target - model learns to maintain style)
        user_pairs = []
        for _, row in user_df.iterrows():
            if len(row['body']) > 50:
                user_pairs.append({
                    'original_body': row['body'],
                    'reply_body': row['body'],
                    'category': row['category']
                })

        if user_pairs:
            user_pair_df = pd.DataFrame(user_pairs)
            df = pd.concat([df, user_pair_df], ignore_index=True)
            print(f"Added {len(user_pairs)} user style examples")
    except FileNotFoundError:
        print("No user emails found - using only Enron (work style)")

    print(f"\nTotal training pairs: {len(df)}")
    print("\nBreakdown by category:")
    print(df['category'].value_counts())

    return df


def train_style_model():
    print("\nTraining style transfer model...")

    # Prepare data
    df = prepare_style_transfer_data()

    # Load vocabulary
    with open('reply_vocabulary.pkl', 'rb') as f:
        vocabulary = pickle.load(f)

    print(f"Vocabulary size: {len(vocabulary.word2idx)}")

    # Split data
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)

    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)}")

    # Create datasets
    train_dataset = StyleTransferDataset(train_df, vocabulary)
    val_dataset = StyleTransferDataset(val_df, vocabulary)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Load existing seq2seq model (reuse architecture)
    from seq2seq_architecture import create_seq2seq_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = create_seq2seq_model(
        vocab_size=len(vocabulary.word2idx),
        embedding_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.3,
        device=device
    )

    # Load pre-trained weights if available
    try:
        checkpoint = torch.load('best_seq2seq_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded pre-trained seq2seq model for fine-tuning")
    except:
        print("Training from scratch")

    # Training
    from train_seq2seq import Seq2SeqTrainer

    trainer = Seq2SeqTrainer(model, train_loader, val_loader, device, learning_rate=0.0001)

    print("\nFine-tuning on style transfer task...")
    history = trainer.train(epochs=15, patience=5)

    # Save style model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocabulary': vocabulary,
    }, 'style_transfer_model.pth')

    print("\nStyle transfer model saved!")

    # Plot results
    from train_seq2seq import plot_losses
    plot_losses(history, 'style_transfer_training.png')

    return model, vocabulary


def test_style_transfer(model, vocabulary, device):
    print("\nTesting style transfer...")

    # Test examples
    test_drafts = [
        "lets meet tomorrow",
        "can you send the report",
        "thanks for the info",
        "meeting at 2pm ok?"
    ]

    model.eval()

    for draft in test_drafts:
        print(f"\nDraft: {draft}")

        src = vocabulary.encode(draft, max_length=100, add_sos_eos=False)
        src_tensor = torch.tensor([src], dtype=torch.long).to(device)

        generated = model.generate(
            src_tensor,
            max_length=50,
            sos_token=2,
            eos_token=3,
            temperature=0.8,
            repetition_penalty=1.5
        )

        rephrased = vocabulary.decode(generated)
        print(f"Rephrased (work style): {rephrased}")


def main():
    model, vocabulary = train_style_model()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_style_transfer(model, vocabulary, device)

    print("\nDone! Use style_transfer_model.pth for personalized rephrasing")


if __name__ == "__main__":
    main()