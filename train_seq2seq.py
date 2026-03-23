import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
import re
from tqdm import tqdm
import matplotlib.pyplot as plt


class ReplyVocabulary:
    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}
        self.word_freq = {}

    def build_vocabulary(self, texts, min_freq=2):
        for text in texts:
            text = text.lower()
            text = re.sub(r'[^a-z0-9\s]', ' ', text)
            tokens = text.split()

            for token in tokens:
                self.word_freq[token] = self.word_freq.get(token, 0) + 1

        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)

        for word, freq in sorted_words[:self.max_vocab_size - 4]:
            if freq >= min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        print(f"Vocabulary size: {len(self.word2idx)}")

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


class EmailReplyDataset(Dataset):
    def __init__(self, df, vocabulary, max_src_len=100, max_trg_len=50):
        self.originals = df['original_body'].tolist()
        self.replies = df['reply_body'].tolist()
        self.vocabulary = vocabulary
        self.max_src_len = max_src_len
        self.max_trg_len = max_trg_len

    def __len__(self):
        return len(self.originals)

    def __getitem__(self, idx):
        src = self.vocabulary.encode(self.originals[idx], self.max_src_len, add_sos_eos=False)
        trg = self.vocabulary.encode(self.replies[idx], self.max_trg_len, add_sos_eos=True)

        return {
            'src': torch.tensor(src, dtype=torch.long),
            'trg': torch.tensor(trg, dtype=torch.long)
        }


class Seq2SeqTrainer:
    def __init__(self, model, train_loader, val_loader, device, learning_rate=0.001):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                              factor=0.5, patience=2)

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, teacher_forcing_ratio=0.5):
        self.model.train()
        epoch_loss = 0

        for batch in tqdm(self.train_loader, desc='Training'):
            src = batch['src'].to(self.device)
            trg = batch['trg'].to(self.device)

            self.optimizer.zero_grad()

            output = self.model(src, trg, teacher_forcing_ratio)

            output_dim = output.shape[-1]
            output = output[:, 1:].contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            loss = self.criterion(output, trg)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        epoch_loss = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                src = batch['src'].to(self.device)
                trg = batch['trg'].to(self.device)

                output = self.model(src, trg, 0)

                output_dim = output.shape[-1]
                output = output[:, 1:].contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)

                loss = self.criterion(output, trg)

                epoch_loss += loss.item()

        return epoch_loss / len(self.val_loader)

    def train(self, epochs=20, patience=5):
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            train_loss = self.train_epoch()
            val_loss = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            self.scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, 'best_seq2seq_model.pth')

                print(f"Model saved with val_loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"Patience: {patience_counter}/{patience}")

                if patience_counter >= patience:
                    print("Early stopping")
                    break

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }


def plot_losses(history, save_path='seq2seq_training.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Seq2Seq Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")


def main():
    print("Loading data...")
    df = pd.read_csv('./data/processed/email_reply_pairs.csv')
    print(f"Loaded {len(df)} pairs")

    print("\nBuilding vocabulary...")
    all_texts = df['original_body'].tolist() + df['reply_body'].tolist()
    vocabulary = ReplyVocabulary(max_vocab_size=15000)
    vocabulary.build_vocabulary(all_texts, min_freq=1)

    with open('reply_vocabulary.pkl', 'wb') as f:
        pickle.dump(vocabulary, f)
    print("Vocabulary saved")

    print("\nSplitting data...")
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    print("\nCreating datasets...")
    train_dataset = EmailReplyDataset(train_df, vocabulary, max_src_len=100, max_trg_len=50)
    val_dataset = EmailReplyDataset(val_df, vocabulary, max_src_len=100, max_trg_len=50)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print("\nCreating model...")
    from seq2seq_architecture import create_seq2seq_model, count_parameters

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

    print(f"Model parameters: {count_parameters(model):,}")

    print("\nTraining...")
    trainer = Seq2SeqTrainer(model, train_loader, val_loader, device, learning_rate=0.001)
    history = trainer.train(epochs=30, patience=8)

    print("\nPlotting results...")
    plot_losses(history)

    print("\nTesting generation...")
    model.load_state_dict(torch.load('best_seq2seq_model.pth')['model_state_dict'])
    model.eval()

    test_original = train_df.iloc[0]['original_body']
    print(f"\nOriginal: {test_original[:100]}...")

    src = torch.tensor([vocabulary.encode(test_original, 100, add_sos_eos=False)]).to(device)
    generated_indices = model.generate(src, max_length=50, sos_token=2, eos_token=3,
                                       temperature=0.8, repetition_penalty=1.5)
    generated_reply = vocabulary.decode(generated_indices)

    print(f"Generated: {generated_reply}")
    print(f"Actual: {train_df.iloc[0]['reply_body'][:100]}...")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()