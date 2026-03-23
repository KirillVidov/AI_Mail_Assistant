import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(Encoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))

        outputs, (hidden, cell) = self.lstm(embedded)

        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        cell = torch.tanh(self.fc(torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)))

        return outputs, hidden, cell


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()

        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        attention = self.v(energy).squeeze(2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim * 2 + embedding_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 3 + embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)

        embedded = self.dropout(self.embedding(input))

        attn_weights = self.attention(hidden, encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)

        weighted = torch.bmm(attn_weights, encoder_outputs)

        rnn_input = torch.cat((embedded, weighted), dim=2)

        hidden_lstm = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell_lstm = cell.unsqueeze(0).repeat(self.num_layers, 1, 1)

        output, (hidden_out, cell_out) = self.lstm(rnn_input, (hidden_lstm, cell_lstm))

        hidden = hidden_out[-1]
        cell = cell_out[-1]

        prediction = self.fc(torch.cat((output.squeeze(1), weighted.squeeze(1), embedded.squeeze(1)), dim=1))

        return prediction, hidden, cell, attn_weights.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)

        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell, attention = self.decoder(input, hidden, cell, encoder_outputs)

            outputs[:, t, :] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)

            input = trg[:, t] if teacher_force else top1

        return outputs

    def generate(self, src, max_length=50, sos_token=2, eos_token=3, temperature=1.0, repetition_penalty=1.2):
        self.eval()

        with torch.no_grad():
            encoder_outputs, hidden, cell = self.encoder(src)

            input = torch.tensor([sos_token]).to(self.device)

            generated = []

            for _ in range(max_length):
                output, hidden, cell, attention = self.decoder(input, hidden, cell, encoder_outputs)

                output = output / temperature

                for token_id in set(generated):
                    output[0, token_id] /= repetition_penalty

                probabilities = torch.softmax(output, dim=1)
                top1 = torch.multinomial(probabilities, 1).item()

                if top1 == eos_token:
                    break

                generated.append(top1)
                input = torch.tensor([top1]).to(self.device)

            return generated


def create_seq2seq_model(vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2, dropout=0.3, device='cpu'):
    encoder = Encoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
    decoder = Decoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)

    model = Seq2Seq(encoder, decoder, device).to(device)

    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    vocab_size = 10000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_seq2seq_model(vocab_size, device=device)

    print(f"Model created with {count_parameters(model):,} trainable parameters")
    print(f"Device: {device}")

    src = torch.randint(0, vocab_size, (2, 20)).to(device)
    trg = torch.randint(0, vocab_size, (2, 15)).to(device)

    output = model(src, trg)
    print(f"Output shape: {output.shape}")

    generated = model.generate(src[0:1])
    print(f"Generated sequence length: {len(generated)}")