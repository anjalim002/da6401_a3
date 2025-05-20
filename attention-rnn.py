# %%
# Cell 1: Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import wandb
from sklearn.model_selection import train_test_split
import os

import wandb
wandb.login(key="5xxxxxxxxxxxxxxxxxxxxxxxxx27c")

# %%
# Cell 2: Dataset class and data loading
class TransliterationDataset(Dataset):
    def __init__(self, latin_texts, devanagari_texts, latin_vocab, devanagari_vocab):
        self.latin_texts = latin_texts
        self.devanagari_texts = devanagari_texts
        self.latin_vocab = latin_vocab
        self.devanagari_vocab = devanagari_vocab
        
    def __len__(self):
        return len(self.latin_texts)
    
    def __getitem__(self, idx):
        latin_text = self.latin_texts[idx]
        devanagari_text = self.devanagari_texts[idx]
        
        # Convert to indices
        latin_indices = [self.latin_vocab.get(char, self.latin_vocab['<UNK>']) for char in latin_text]
        devanagari_indices = [self.devanagari_vocab.get(char, self.devanagari_vocab['<UNK>']) for char in devanagari_text]
        
        # Add EOS token
        latin_indices.append(self.latin_vocab['<EOS>'])
        devanagari_indices.append(self.devanagari_vocab['<EOS>'])
        
        return {
            'latin': torch.tensor(latin_indices, dtype=torch.long),
            'devanagari': torch.tensor(devanagari_indices, dtype=torch.long),
            'latin_len': len(latin_indices),
            'devanagari_len': len(devanagari_indices),
            'latin_text': latin_text,
            'devanagari_text': devanagari_text
        }

def collate_fn(batch):
    latin_lens = [item['latin_len'] for item in batch]
    devanagari_lens = [item['devanagari_len'] for item in batch]
    
    max_latin_len = max(latin_lens)
    max_devanagari_len = max(devanagari_lens)
    
    latin_padded = torch.zeros((len(batch), max_latin_len), dtype=torch.long)
    devanagari_padded = torch.zeros((len(batch), max_devanagari_len), dtype=torch.long)
    
    latin_texts = []
    devanagari_texts = []
    
    for i, item in enumerate(batch):
        latin = item['latin']
        devanagari = item['devanagari']
        
        latin_padded[i, :latin.size(0)] = latin
        devanagari_padded[i, :devanagari.size(0)] = devanagari
        
        latin_texts.append(item['latin_text'])
        devanagari_texts.append(item['devanagari_text'])
    
    return {
        'latin': latin_padded,
        'devanagari': devanagari_padded,
        'latin_lens': torch.tensor(latin_lens),
        'devanagari_lens': torch.tensor(devanagari_lens),
        'latin_texts': latin_texts,
        'devanagari_texts': devanagari_texts
    }
def load_data(train_path, dev_path, test_path):
    # Load only the first two relevant columns: 0 (output), 1 (input)
    train_df = pd.read_csv(train_path, sep='\t', header=None, usecols=[0, 1])
    dev_df = pd.read_csv(dev_path, sep='\t', header=None, usecols=[0, 1])
    test_df = pd.read_csv(test_path, sep='\t', header=None, usecols=[0, 1])

    train_df.dropna(inplace=True)
    dev_df.dropna(inplace=True)
    test_df.dropna(inplace=True)
    
    # Extract input (latin) and output (devanagari) texts
    train_devanagari, train_latin = train_df[0].tolist(), train_df[1].tolist()
    dev_devanagari, dev_latin = dev_df[0].tolist(), dev_df[1].tolist()
    test_devanagari, test_latin = test_df[0].tolist(), test_df[1].tolist()
    
    # Build vocabularies
    latin_chars = set()
    devanagari_chars = set()
    
    for text in train_latin + dev_latin + test_latin:
        latin_chars.update(str(text))  # handle any NaN
    
    for text in train_devanagari + dev_devanagari + test_devanagari:
        devanagari_chars.update(str(text))  # handle any NaN
    
    # Create vocabularies with special tokens
    latin_vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    for i, char in enumerate(sorted(latin_chars)):
        latin_vocab[char] = i + 4
    
    devanagari_vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    for i, char in enumerate(sorted(devanagari_chars)):
        devanagari_vocab[char] = i + 4
    
    # Create inverse vocabularies for decoding
    latin_inv_vocab = {v: k for k, v in latin_vocab.items()}
    devanagari_inv_vocab = {v: k for k, v in devanagari_vocab.items()}
    
    # Create datasets
    train_dataset = TransliterationDataset(train_latin, train_devanagari, latin_vocab, devanagari_vocab)
    dev_dataset = TransliterationDataset(dev_latin, dev_devanagari, latin_vocab, devanagari_vocab)
    test_dataset = TransliterationDataset(test_latin, test_devanagari, latin_vocab, devanagari_vocab)
    
    return {
        'train_dataset': train_dataset,
        'dev_dataset': dev_dataset,
        'test_dataset': test_dataset,
        'latin_vocab': latin_vocab,
        'devanagari_vocab': devanagari_vocab,
        'latin_inv_vocab': latin_inv_vocab,
        'devanagari_inv_vocab': devanagari_inv_vocab
    }


# %%
# Cell 3: Attention-based Seq2Seq model
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout, cell_type):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.cell_type = cell_type
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
        if cell_type == 'RNN':
            self.rnn = nn.RNN(emb_dim, hidden_dim, n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True)
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True)
        
    def forward(self, src, src_lens):
        # src: [batch_size, src_len]
        
        embedded = self.dropout_layer(self.embedding(src))
        # embedded: [batch_size, src_len, emb_dim]
        
        # Pack padded sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        
        if self.cell_type == 'LSTM':
            packed_outputs, (hidden, cell) = self.rnn(packed_embedded)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
            return outputs, hidden, cell
        else:
            packed_outputs, hidden = self.rnn(packed_embedded)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
            return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden: [batch_size, hidden_dim]
        # encoder_outputs: [batch_size, src_len, hidden_dim]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repeat hidden for each word in the source
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # hidden: [batch_size, src_len, hidden_dim]
        
        # Concatenate encoder outputs and hidden state
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy: [batch_size, src_len, hidden_dim]
        
        attention = self.v(energy).squeeze(2)
        # attention: [batch_size, src_len]
        
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout, cell_type, attention):
        super().__init__()
        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.cell_type = cell_type
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
        if cell_type == 'RNN':
            self.rnn = nn.RNN(emb_dim + hidden_dim, hidden_dim, n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True)
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(emb_dim + hidden_dim, hidden_dim, n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(emb_dim + hidden_dim, hidden_dim, n_layers, dropout=dropout if n_layers > 1 else 0, batch_first=True)
        
        self.fc_out = nn.Linear(hidden_dim * 2 + emb_dim, output_dim)
        
    def forward(self, input, hidden, encoder_outputs, cell=None, mask=None):
        # input: [batch_size, 1]
        # hidden: [n_layers, batch_size, hidden_dim]
        # encoder_outputs: [batch_size, src_len, hidden_dim]
        
        input = input.unsqueeze(1)  # [batch_size, 1]
        
        embedded = self.dropout_layer(self.embedding(input))
        # embedded: [batch_size, 1, emb_dim]
        
        # Get attention weights
        if self.cell_type == 'LSTM':
            # For LSTM, use the top layer hidden state for attention
            attn_weights = self.attention(hidden[-1], encoder_outputs, mask)
        else:
            attn_weights = self.attention(hidden[-1], encoder_outputs, mask)
        
        # attn_weights: [batch_size, src_len]
        
        # Create context vector by multiplying attention weights with encoder outputs
        attn_weights = attn_weights.unsqueeze(1)
        # attn_weights: [batch_size, 1, src_len]
        
        context = torch.bmm(attn_weights, encoder_outputs)
        # context: [batch_size, 1, hidden_dim]
        
        # Combine embedded input and context vector
        rnn_input = torch.cat((embedded, context), dim=2)
        # rnn_input: [batch_size, 1, emb_dim + hidden_dim]
        
        if self.cell_type == 'LSTM':
            output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
            # output: [batch_size, 1, hidden_dim]
        else:
            output, hidden = self.rnn(rnn_input, hidden)
            # output: [batch_size, 1, hidden_dim]
        
        # Squeeze the sequence length dimension
        output = output.squeeze(1)
        context = context.squeeze(1)
        embedded = embedded.squeeze(1)
        
        # Make prediction
        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))
        # prediction: [batch_size, output_dim]
        
        if self.cell_type == 'LSTM':
            return prediction, hidden, cell, attn_weights
        else:
            return prediction, hidden, attn_weights

class Seq2SeqWithAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hidden_dim == decoder.hidden_dim, "Hidden dimensions of encoder and decoder must be equal!"
        
    def create_mask(self, src, src_lens):
        # src: [batch_size, src_len]
        batch_size = src.shape[0]
        max_src_len = src.shape[1]
        
        # Create a mask of shape [batch_size, src_len]
        mask = torch.zeros(batch_size, max_src_len, device=self.device)
        
        for i, length in enumerate(src_lens):
            mask[i, :length] = 1
            
        return mask
    
    def forward(self, src, src_lens, trg, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Tensor to store attention weights
        attentions = torch.zeros(batch_size, trg_len, src.shape[1]).to(self.device)
        
        # Create a mask for encoder outputs
        mask = self.create_mask(src, src_lens)
        
        # Encode the source sequence
        if self.encoder.cell_type == 'LSTM':
            encoder_outputs, hidden, cell = self.encoder(src, src_lens)
        else:
            encoder_outputs, hidden = self.encoder(src, src_lens)
            cell = None
        
        # First input to the decoder is the <SOS> token
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            # Decode one step
            if self.decoder.cell_type == 'LSTM':
                output, hidden, cell, attn_weights = self.decoder(input, hidden, encoder_outputs, cell, mask)
            else:
                output, hidden, attn_weights = self.decoder(input, hidden, encoder_outputs, mask=mask)
            
            # Store output and attention
            outputs[:, t, :] = output
            attentions[:, t, :] = attn_weights.squeeze(1)
            
            # Teacher forcing: use real target as next input or use predicted token
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            
            input = trg[:, t] if teacher_force else top1
            
        return outputs, attentions

# %%
# Cell 4: Training, evaluation, and testing functions
def train(model, iterator, optimizer, criterion, teacher_forcing_ratio, clip, device):
    model.train()
    
    epoch_loss = 0
    
    for batch in iterator:
        src = batch['latin'].to(device)
        trg = batch['devanagari'].to(device)
        src_lens = batch['latin_lens']
        
        optimizer.zero_grad()
        
        output, _ = model(src, src_lens, trg, teacher_forcing_ratio)
        
        # Exclude the first token (SOS) from the loss calculation
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    model.eval()
    
    epoch_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in iterator:
            src = batch['latin'].to(device)
            trg = batch['devanagari'].to(device)
            src_lens = batch['latin_lens']
            
            output, _ = model(src, src_lens, trg, 0)  # No teacher forcing during evaluation
            
            # Exclude the first token (SOS) from the loss calculation
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
            
            # Calculate sequence accuracy
            output = output.view(batch['devanagari'].shape[0], -1, output_dim)
            predictions = output.argmax(2)
            
            # Compare predictions with targets (excluding padding)
            for i in range(len(batch['devanagari_texts'])):
                pred_seq = []
                for j in range(1, batch['devanagari_lens'][i] - 1):  # Exclude SOS and EOS
                    pred_token = predictions[i, j-1].item()
                    if pred_token != 0:  # Not PAD
                        pred_seq.append(pred_token)
                
                true_seq = []
                for j in range(1, batch['devanagari_lens'][i] - 1):  # Exclude SOS and EOS
                    true_token = trg[i * (batch['devanagari_lens'].max().item() - 1) + (j-1)].item()
                    if true_token != 0:  # Not PAD
                        true_seq.append(true_token)
                
                if len(pred_seq) == len(true_seq) and all(p == t for p, t in zip(pred_seq, true_seq)):
                    correct_predictions += 1
                    
                total_predictions += 1
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    return epoch_loss / len(iterator), accuracy

def translate_sentence(sentence, src_vocab, trg_vocab, model, device, max_len=100):
    model.eval()
    
    # Convert sentence to indices
    tokens = [src_vocab.get(char, src_vocab['<UNK>']) for char in sentence]
    tokens.append(src_vocab['<EOS>'])
    
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    src_lens = torch.LongTensor([len(tokens)])
    
    # Create mask for encoder outputs
    mask = model.create_mask(src_tensor, src_lens)
    
    with torch.no_grad():
        if model.encoder.cell_type == 'LSTM':
            encoder_outputs, hidden, cell = model.encoder(src_tensor, src_lens)
        else:
            encoder_outputs, hidden = model.encoder(src_tensor, src_lens)
            cell = None
    
    # First token is SOS
    trg_idx = [trg_vocab['<SOS>']]
    attentions = []
    
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_idx[-1]]).to(device)
        
        with torch.no_grad():
            if model.decoder.cell_type == 'LSTM':
                output, hidden, cell, attention = model.decoder(trg_tensor, hidden, encoder_outputs, cell, mask)
            else:
                output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask=mask)
                
        attentions.append(attention.squeeze(1).cpu().numpy())
        
        pred_token = output.argmax(1).item()
        
        # Stop if EOS
        if pred_token == trg_vocab['<EOS>']:
            break
        
        trg_idx.append(pred_token)
    
    # Convert indices to tokens
    trg_tokens = [trg_vocab[i] for i in range(len(trg_vocab)) if i in trg_idx]
    
    # Remove special tokens
    trg_tokens = [token for token in trg_tokens if token not in ['<SOS>', '<EOS>', '<PAD>', '<UNK>']]
    
    return ''.join(trg_tokens), attentions

def calculate_sequence_accuracy(predictions, targets):
    correct = 0
    total = len(targets)
    
    for pred, target in zip(predictions, targets):
        if pred == target:
            correct += 1
    
    return correct / total

# %%
# # Cell 3 correction: Fix translate_sentence function
# def translate_sentence(sentence, src_vocab, trg_vocab, model, device, max_len=100):
#     model.eval()
    
#     # Convert sentence to indices
#     tokens = [src_vocab.get(char, src_vocab['<UNK>']) for char in sentence]
#     tokens.append(src_vocab['<EOS>'])
    
#     src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
#     src_lens = torch.LongTensor([len(tokens)])
    
#     # Create mask for encoder outputs
#     mask = model.create_mask(src_tensor, src_lens)
    
#     with torch.no_grad():
#         if model.encoder.cell_type == 'LSTM':
#             encoder_outputs, hidden, cell = model.encoder(src_tensor, src_lens)
#         else:
#             encoder_outputs, hidden = model.encoder(src_tensor, src_lens)
#             cell = None
    
#     # First token is SOS
#     trg_idx = [trg_vocab['< SOS >']]
#     attentions = []
    
#     for i in range(max_len):
#         trg_tensor = torch.LongTensor([trg_idx[-1]]).to(device)
        
#         with torch.no_grad():
#             if model.decoder.cell_type == 'LSTM':
#                 output, hidden, cell, attention = model.decoder(trg_tensor, hidden, encoder_outputs, cell, mask)
#             else:
#                 output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask=mask)
                
#         attentions.append(attention.squeeze(1).cpu().numpy())
        
#         pred_token = output.argmax(1).item()
        
#         # Stop if EOS
#         if pred_token == trg_vocab['<EOS>']:
#             break
        
#         trg_idx.append(pred_token)
    
#     # Create inverse vocab dictionary if not provided
#     trg_inv_vocab = {v: k for k, v in trg_vocab.items()}
    
#     # Convert indices to tokens using inverse vocab
#     trg_tokens = [trg_inv_vocab.get(idx, '<UNK>') for idx in trg_idx]
    
#     # Remove special tokens
#     trg_tokens = [token for token in trg_tokens if token not in ['< SOS >', '<EOS>', '<PAD>', '<UNK>']]
    
#     return ''.join(trg_tokens), attentions

def translate_sentence(sentence, src_vocab, trg_vocab, model, device, max_len=100):
    model.eval()
    
    # Convert sentence to indices
    tokens = [src_vocab.get(char, src_vocab['<UNK>']) for char in sentence]
    tokens.append(src_vocab['<EOS>'])
    
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    src_lens = torch.LongTensor([len(tokens)])
    
    # Create mask for encoder outputs
    mask = model.create_mask(src_tensor, src_lens)
    
    with torch.no_grad():
        if model.encoder.cell_type == 'LSTM':
            encoder_outputs, hidden, cell = model.encoder(src_tensor, src_lens)
        else:
            encoder_outputs, hidden = model.encoder(src_tensor, src_lens)
            cell = None
    
    # First token is SOS
    trg_idx = [trg_vocab['<SOS>']]
    attentions = []
    
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_idx[-1]]).to(device)
        
        with torch.no_grad():
            if model.decoder.cell_type == 'LSTM':
                output, hidden, cell, attention = model.decoder(trg_tensor, hidden, encoder_outputs, cell, mask)
            else:
                output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask=mask)
                
        attentions.append(attention.squeeze(1).cpu().numpy())
        
        pred_token = output.argmax(1).item()
        
        # Stop if EOS
        if pred_token == trg_vocab['<EOS>']:
            break
        
        trg_idx.append(pred_token)
    
    results = load_data(train_path='/kaggle/input/hindi-dl/hi.translit.sampled.train.tsv', dev_path='/kaggle/input/hindi-dl/hi.translit.sampled.dev.tsv', test_path='/kaggle/input/hindi-dl/hi.translit.sampled.test.tsv')
    # Convert indices to tokens
    trg_inv_vocab = {v: k for k, v in trg_vocab.items()}
    trg_tokens = [results['devanagari_inv_vocab'].get(i, '<UNK>') for i in trg_idx]
    
    # Remove special tokens
    trg_tokens = [token for token in trg_tokens if token not in ['<SOS>', '<EOS>', '<PAD>', '<UNK>']]
    
    return ''.join(trg_tokens), attentions

# %%
# Cell 4: Training, evaluation, and testing functions
def train(model, iterator, optimizer, criterion, teacher_forcing_ratio, clip, device):
    model.train()
    
    epoch_loss = 0
    
    for batch in iterator:
        src = batch['latin'].to(device)
        trg = batch['devanagari'].to(device)
        src_lens = batch['latin_lens']
        
        optimizer.zero_grad()
        
        output, _ = model(src, src_lens, trg, teacher_forcing_ratio)
        
        # Exclude the first token (SOS) from the loss calculation
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    model.eval()
    
    epoch_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in iterator:
            src = batch['latin'].to(device)
            trg = batch['devanagari'].to(device)
            src_lens = batch['latin_lens']
            
            output, _ = model(src, src_lens, trg, 0)  # No teacher forcing during evaluation
            
            # Exclude the first token (SOS) from the loss calculation
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
            
            # Calculate sequence accuracy
            output = output.view(batch['devanagari'].shape[0], -1, output_dim)
            predictions = output.argmax(2)
            
            # Compare predictions with targets (excluding padding)
            for i in range(len(batch['devanagari_texts'])):
                pred_seq = []
                for j in range(1, batch['devanagari_lens'][i] - 1):  # Exclude SOS and EOS
                    pred_token = predictions[i, j-1].item()
                    if pred_token != 0:  # Not PAD
                        pred_seq.append(pred_token)
                
                true_seq = []
                for j in range(1, batch['devanagari_lens'][i] - 1):  # Exclude SOS and EOS
                    true_token = trg[i * (batch['devanagari_lens'].max().item() - 1) + (j-1)].item()
                    if true_token != 0:  # Not PAD
                        true_seq.append(true_token)
                
                if len(pred_seq) == len(true_seq) and all(p == t for p, t in zip(pred_seq, true_seq)):
                    correct_predictions += 1
                    
                total_predictions += 1
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    return epoch_loss / len(iterator), accuracy

# def translate_sentence(sentence, src_vocab, trg_vocab, model, device, max_len=100):
#     model.eval()
    
#     # Convert sentence to indices
#     tokens = [src_vocab.get(char, src_vocab['<UNK>']) for char in sentence]
#     tokens.append(src_vocab['<EOS>'])
    
#     src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
#     src_lens = torch.LongTensor([len(tokens)])
    
#     # Create mask for encoder outputs
#     mask = model.create_mask(src_tensor, src_lens)
    
#     with torch.no_grad():
#         if model.encoder.cell_type == 'LSTM':
#             encoder_outputs, hidden, cell = model.encoder(src_tensor, src_lens)
#         else:
#             encoder_outputs, hidden = model.encoder(src_tensor, src_lens)
#             cell = None
    
    # # First token is SOS
    # trg_idx = [trg_vocab['<SOS>']]
    # attentions = []
    
    # for i in range(max_len):
    #     trg_tensor = torch.LongTensor([trg_idx[-1]]).to(device)
        
    #     with torch.no_grad():
    #         if model.decoder.cell_type == 'LSTM':
    #             output, hidden, cell, attention = model.decoder(trg_tensor, hidden, encoder_outputs, cell, mask)
    #         else:
    #             output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask=mask)
                
    #     attentions.append(attention.squeeze(1).cpu().numpy())
        
    #     pred_token = output.argmax(1).item()
        
    #     # Stop if EOS
    #     if pred_token == trg_vocab['<EOS>']:
    #         break
        
    #     trg_idx.append(pred_token)
    
    # Convert indices to tokens
    # trg_tokens = [trg_vocab[i] for i in range(len(trg_vocab)) if i in trg_idx]
    # trg_tokens = [trg_inv_vocab.get(i, '<UNK>') for i in trg_idx]

    # # Remove special tokens
    # trg_tokens = [token for token in trg_tokens if token not in ['<SOS>', '<EOS>', '<PAD>', '<UNK>']]
    
    # return ''.join(trg_tokens), attentions

def calculate_sequence_accuracy(predictions, targets):
    correct = 0
    total = len(targets)
    
    for pred, target in zip(predictions, targets):
        if pred == target:
            correct += 1
    
    return correct / total

# %%
# Cell 5: WandB sweep function
def wandb_sweep_train():
    # Initialize wandb
    wandb.init()
    
    # Get hyperparameters from wandb
    config = wandb.config
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    data = load_data('/kaggle/input/hindi-dl/hi.translit.sampled.train.tsv', 
                    '/kaggle/input/hindi-dl/hi.translit.sampled.dev.tsv', 
                    '/kaggle/input/hindi-dl/hi.translit.sampled.test.tsv')
    
    train_dataset = data['train_dataset']
    dev_dataset = data['dev_dataset']
    latin_vocab = data['latin_vocab']
    devanagari_vocab = data['devanagari_vocab']
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, collate_fn=collate_fn)
    
    # Set model parameters
    input_dim = len(latin_vocab)
    output_dim = len(devanagari_vocab)
    
    # Initialize attention
    attention = Attention(config.hidden_dim)
    
    # Initialize encoder and decoder
    encoder = Encoder(input_dim, config.emb_dim, config.hidden_dim, config.enc_layers, config.dropout, config.cell_type)
    decoder = Decoder(output_dim, config.emb_dim, config.hidden_dim, config.dec_layers, config.dropout, config.cell_type, attention)
    
    # Initialize the model
    model = Seq2SeqWithAttention(encoder, decoder, device).to(device)
    
    # Print model parameters
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
    
    # Initialize optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    # Initialize early stopping variables
    best_valid_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.epochs):
        # Train
        train_loss = train(model, train_loader, optimizer, criterion, config.teacher_forcing, 1.0, device)
        
        # Evaluate
        valid_loss, valid_acc = evaluate(model, dev_loader, criterion, device)
        
        # Log metrics
        wandb.log({
            'train_loss': train_loss,
            'val_loss': valid_loss,
            'val_sequence_accuracy': valid_acc,
            'epoch': epoch
        })
        
        print(f'Epoch: {epoch+1}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tValid Loss: {valid_loss:.3f}')
        print(f'\tValid Acc: {valid_acc:.3f}')
        
        # Early stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            
        if patience_counter >= config.patience:
            print("Early stopping!")
            break
    
    return model

# %%
# Cell 6: Main execution code
def main():
    # Initialize wandb
    wandb.login()
    
    # Define sweep configuration
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'val_sequence_accuracy', 'goal': 'maximize'},
        'parameters': {
            'emb_dim': {'values': [64, 128, 256]},
            'hidden_dim': {'values': [128, 256]},
            'enc_layers': {'values': [1, 2, 3]},
            'dec_layers': {'values': [1, 2, 3]},
            'cell_type': {'values': ['LSTM', 'GRU', 'RNN']},
            'dropout': {'values': [0.2, 0.3, 0.4]},
            'batch_size': {'values': [32, 64, 128]},
            'learning_rate': {'values': [0.001, 0.0005, 0.0001]},
            'teacher_forcing': {'values': [0.5, 0.7, 0.9]},
            'patience': {'value': 3},
            'epochs': {'values': [10, 15]}
        }
    }
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project="hindi-transliteration-attention")
    
    # Run sweep
    wandb.agent(sweep_id, wandb_sweep_train, count=15)  # Run 15 experiments

# Uncomment to run the sweep
# if __name__ == "__main__":
#     main()

# %%

def evaluate_best_model():
    # Load data
    data = load_data('/kaggle/input/hindi-dl/hi.translit.sampled.train.tsv', 
                     '/kaggle/input/hindi-dl/hi.translit.sampled.dev.tsv', 
                     '/kaggle/input/hindi-dl/hi.translit.sampled.test.tsv')
 
    test_dataset = data['test_dataset']
    latin_vocab = data['latin_vocab']
    devanagari_vocab = data['devanagari_vocab']
    latin_inv_vocab = data['latin_inv_vocab']
    devanagari_inv_vocab = data['devanagari_inv_vocab']
    
    # Create dataloaders
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the best hyperparameters (assume these are from the sweep)
    best_config = {
        'emb_dim': 256,
        'hidden_dim': 256,
        'enc_layers': 2,
        'dec_layers': 2,
        'cell_type': 'LSTM',
        'dropout': 0.3
    }
    
    # Set model parameters
    input_dim = len(latin_vocab)
    output_dim = len(devanagari_vocab)
    
    # Initialize attention
    attention = Attention(best_config['hidden_dim'])
    
    # Initialize encoder and decoder
    encoder = Encoder(input_dim, best_config['emb_dim'], best_config['hidden_dim'], 
                     best_config['enc_layers'], best_config['dropout'], best_config['cell_type'])
    decoder = Decoder(output_dim, best_config['emb_dim'], best_config['hidden_dim'], 
                     best_config['dec_layers'], best_config['dropout'], best_config['cell_type'], attention)
    
    # Initialize the model
    model = Seq2SeqWithAttention(encoder, decoder, device).to(device)
    
    # Load the best model
    model.load_state_dict(torch.load('best_model.pt'))
    
    # Evaluate on test set
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    
    print(f'Test Loss: {test_loss:.3f}')
    print(f'Test Accuracy: {test_accuracy:.3f}')
    
    # Generate predictions for test samples
    all_predictions = []
    all_targets = []
    all_inputs = []
    attention_maps = []
    
    # Get a sample of test data for visualization
    test_samples = []
    for i, item in enumerate(test_dataset):
        if i < 9:  # Get 9 samples for visualization
            latin_text = item['latin_text']
            devanagari_text = item['devanagari_text']
            test_samples.append((latin_text, devanagari_text))
    
    # Generate predictions and attention maps
    for latin_text, devanagari_text in test_samples:
        prediction, attentions = translate_sentence(latin_text, latin_vocab, devanagari_vocab, model, device)
        
        all_inputs.append(latin_text)
        all_predictions.append(prediction)
        all_targets.append(devanagari_text)
        attention_maps.append((latin_text, prediction, devanagari_text, attentions))
    
    # Save predictions to CSV file
    import csv
    with open('predictions_attention.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['Input', 'Prediction', 'Target'])
        # Write data
        for inp, pred, target in zip(all_inputs, all_predictions, all_targets):
            writer.writerow([inp, pred, target])
    
    # Visualize attention maps
    visualize_attention_maps(attention_maps)
    
    return test_accuracy, all_predictions, all_targets

def visualize_attention_maps(attention_maps):
    """
    Visualize attention maps for test samples.
    
    Args:
        attention_maps: List of tuples (latin_text, prediction, devanagari_text, attentions)
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    for i, (latin_text, prediction, devanagari_text, attentions) in enumerate(attention_maps[:9]):
        row = i // 3
        col = i % 3
        
        # Get attention for the first token (we'll just use the average of all attentions)
        attention = np.mean(np.array(attentions), axis=0)
        
        # Plot heatmap
        sns.heatmap(attention, xticklabels=list(latin_text) + ['<EOS>'], 
                   yticklabels=['<AVG>'], ax=axes[row, col], cmap='viridis')
        
        axes[row, col].set_title(f"Input: {latin_text}\nPred: {prediction}\nTarget: {devanagari_text}")
    
    plt.tight_layout()
    plt.savefig('attention_maps.png')
    plt.close()

# %%
def compare_models():
    """Compare attention-based model with vanilla model."""
    import csv
    
    # Try to load vanilla model predictions
    try:
        vanilla_preds = []
        vanilla_targets = []
        attention_preds = []
        attention_targets = []
        inputs = []
        
        # Parse vanilla predictions
        with open('/kaggle/input/prediction-vanilla/test_predictions.csv', 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader, None)  # Skip header if exists
            for row in csv_reader:
                if len(row) >= 3:  # Assuming CSV format: input, prediction, target
                    inputs.append(row[0])
                    vanilla_preds.append(row[1])
                    vanilla_targets.append(row[2])
        
        # Parse attention predictions
        with open('/kaggle/working/predictions_attention.csv', 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader, None)  # Skip header if exists
            for row in csv_reader:
                if len(row) >= 3:  # Assuming CSV format: input, prediction, target
                    attention_preds.append(row[1])
                    attention_targets.append(row[2])
        
        # Calculate accuracies
        vanilla_correct = sum(1 for p, t in zip(vanilla_preds, vanilla_targets) if p == t)
        attention_correct = sum(1 for p, t in zip(attention_preds, attention_targets) if p == t)
        
        vanilla_acc = vanilla_correct / len(vanilla_targets) if vanilla_targets else 0
        attention_acc = attention_correct / len(attention_targets) if attention_targets else 0
        
        print(f"Vanilla Accuracy: {vanilla_acc:.4f}")
        print(f"Attention Accuracy: {attention_acc:.4f}")
        
        # Find examples that attention model corrects
        corrected_examples = []
        for i, (v_pred, a_pred, target, inp) in enumerate(zip(vanilla_preds, attention_preds, attention_targets, inputs)):
            if v_pred != target and a_pred == target:
                corrected_examples.append((inp, v_pred, a_pred, target))
        
        # print(f"\nExamples corrected by attention model ({len(corrected_examples)}):")
        for i, (inp, v_pred, a_pred, target) in enumerate(corrected_examples[:5]):  # Show at most 5 examples
            # print(f"Example {i+1}:")
            # print(f"  Input: {inp}")
            # print(f"  Vanilla prediction: {v_pred}")
            # print(f"  Attention prediction: {a_pred}")
            # print(f"  Target: {target}")
            print()
        
        return vanilla_acc, attention_acc, corrected_examples
    except FileNotFoundError:
        print()
        return None, None, None

# %%
# Cell 9: Function to run a single experiment
def run_single_experiment():
    """Run a single experiment with best hyperparameters."""
    # Initialize wandb
    wandb.init(project="hindi-transliteration-attention", name="single_experiment")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    data = load_data('/kaggle/input/hindi-dl/hi.translit.sampled.train.tsv', 
                    '/kaggle/input/hindi-dl/hi.translit.sampled.dev.tsv', 
                    '/kaggle/input/hindi-dl/hi.translit.sampled.test.tsv')
    
    
    train_dataset = data['train_dataset']
    dev_dataset = data['dev_dataset']
    latin_vocab = data['latin_vocab']
    devanagari_vocab = data['devanagari_vocab']
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    
    # Set model parameters
    input_dim = len(latin_vocab)
    output_dim = len(devanagari_vocab)
    
    # Best hyperparameters
    config = {
        'emb_dim': 256,
        'hidden_dim': 256,
        'enc_layers': 2,
        'dec_layers': 2,
        'cell_type': 'LSTM',
        'dropout': 0.3,
        'learning_rate': 0.001,
        'teacher_forcing': 0.7,
        'patience': 3,
        'epochs': 3
    }
    
    # Initialize attention
    attention = Attention(config['hidden_dim'])
    
    # Initialize encoder and decoder
    encoder = Encoder(input_dim, config['emb_dim'], config['hidden_dim'], 
                     config['enc_layers'], config['dropout'], config['cell_type'])
    decoder = Decoder(output_dim, config['emb_dim'], config['hidden_dim'], 
                     config['dec_layers'], config['dropout'], config['cell_type'], attention)
    
    # Initialize the model
    model = Seq2SeqWithAttention(encoder, decoder, device).to(device)
    
    # Print model parameters
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
    
    # Initialize optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    # Initialize early stopping variables
    best_valid_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        # Train
        train_loss = train(model, train_loader, optimizer, criterion, config['teacher_forcing'], 1.0, device)
        
        # Evaluate
        valid_loss, valid_acc = evaluate(model, dev_loader, criterion, device)
        
        # Log metrics
        wandb.log({
            'train_loss': train_loss,
            'val_loss': valid_loss,
            'val_sequence_accuracy': valid_acc,
            'epoch': epoch
        })
        
        print(f'Epoch: {epoch+1}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tValid Loss: {valid_loss:.3f}')
        print(f'\tValid Acc: {valid_acc:.3f}')
        
        # Early stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            
        if patience_counter >= config['patience']:
            print("Early stopping!")
            break
    
    return model

# %%
# Cell 10: Main function
def main():
    # Choose whether to run sweep or single experiment
    run_sweep = False  # Set to True to run sweep, False to run single experiment
    
    if run_sweep:
        print("Running hyperparameter sweep...")
        # Initialize wandb
        wandb.login()
        
        # Define sweep configuration
        sweep_config = {
            'method': 'bayes',
            'metric': {'name': 'val_sequence_accuracy', 'goal': 'maximize'},
            'parameters': {
                'emb_dim': {'values': [64, 128, 256]},
                'hidden_dim': {'values': [128, 256]},
                'enc_layers': {'values': [1, 2, 3]},
                'dec_layers': {'values': [1, 2, 3]},
                'cell_type': {'values': ['LSTM', 'GRU', 'RNN']},
                'dropout': {'values': [0.2, 0.3, 0.4]},
                'batch_size': {'values': [32, 64, 128]},
                'learning_rate': {'values': [0.001, 0.0005, 0.0001]},
                'teacher_forcing': {'values': [0.5, 0.7, 0.9]},
                'patience': {'value': 3},
                'epochs': {'values': [10, 15]}
            }
        }
        
        # Create sweep
        sweep_id = wandb.sweep(sweep_config, project="hindi-transliteration-attention")
        
        # Run sweep
        wandb.agent(sweep_id, wandb_sweep_train, count=15)  # Run 15 experiments
    else:
        print("Running single experiment...")
        # Run single experiment
        model = run_single_experiment()
    
    # Evaluate best model
    print("\nEvaluating best model on test set...")
    accuracy, predictions, targets = evaluate_best_model()
    
    # Compare with vanilla model
    print("\nComparing with vanilla model...")
    vanilla_acc, attention_acc, corrected_examples = compare_models()
    
    print(f"\nAttention Model Test Accuracy: {accuracy:.4f}")
    if vanilla_acc is not None:
        print(f"Improvement over Vanilla Model: {(attention_acc - vanilla_acc)*100:.2f}%")
    
    # Display WandB report link
    print("\nCheck WandB for detailed reports and visualizations.")

if __name__ == "__main__":
    main()

# %%
# Cell 11: Connectivity visualization (Q6)
def plot_connectivity(input_sentence, output_sentence, attention_weights):
    """
    Plot connectivity visualization as mentioned in Q6.
    This shows which input character the model is attending to when decoding each output character.
    
    Args:
        input_sentence: Input string in Latin
        output_sentence: Output string in Devanagari
        attention_weights: Attention weights from the model
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Remove first attention weight (for SOS token)
    attention_weights = np.concatenate(attention_weights[1:], axis=0)
    
    # Get the maximum attention weight for each output character
    max_attention_indices = np.argmax(attention_weights, axis=1)
    
    # Plot input and output characters
    input_x = np.arange(len(input_sentence))
    output_y = np.arange(len(output_sentence))
    
    # Plot input characters
    for i, char in enumerate(input_sentence):
        plt.text(i, -0.2, char, fontsize=14, ha='center')
    
    # Plot output characters
    for i, char in enumerate(output_sentence):
        plt.text(-0.5, i, char, fontsize=14, va='center')
    
    # Plot connections
    for i in range(len(output_sentence)):
        j = max_attention_indices[i]
        weight = attention_weights[i, j]
        
        # Draw line with alpha proportional to weight
        plt.plot([j, -0.2], [i, i], 'r-', alpha=weight, linewidth=2)
        
    # Set axis limits
    plt.xlim(-1, len(input_sentence))
    plt.ylim(-1, len(output_sentence))
    
    # Remove axis ticks
    plt.xticks([])
    plt.yticks([])
    
    # Add title
    plt.title('Connectivity: Which input character is attended to when decoding each output character?')
    
    # Save figure
    plt.savefig('predictions_attention/connectivity_visualization.png')
    plt.close()

def generate_connectivity_visualizations():
    """
    Generate connectivity visualizations for 5 examples
    """
    # Load data
    data = load_data('/kaggle/input/hindi-dl/hi.translit.sampled.train.tsv', 
                    '/kaggle/input/hindi-dl/hi.translit.sampled.dev.tsv', 
                    '/kaggle/input/hindi-dl/hi.translit.sampled.test.tsv')
    
    test_dataset = data['test_dataset']
    latin_vocab = data['latin_vocab']
    devanagari_vocab = data['devanagari_vocab']
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model configuration
    config = {
        'emb_dim': 256,
        'hidden_dim': 256,
        'enc_layers': 2,
        'dec_layers': 2,
        'cell_type': 'LSTM',
        'dropout': 0.3
    }
    
    # Set model parameters
    input_dim = len(latin_vocab)
    output_dim = len(devanagari_vocab)
    
    # Initialize attention
    attention = Attention(config['hidden_dim'])
    
    # Initialize encoder and decoder
    encoder = Encoder(input_dim, config['emb_dim'], config['hidden_dim'], 
                     config['enc_layers'], config['dropout'], config['cell_type'])
    decoder = Decoder(output_dim, config['emb_dim'], config['hidden_dim'], 
                     config['dec_layers'], config['dropout'], config['cell_type'], attention)
    
    # Initialize the model
    model = Seq2SeqWithAttention(encoder, decoder, device).to(device)
    
    # Load the best model
    model.load_state_dict(torch.load('best_model.pt'))
    
    # Sample 5 examples for visualization
    sample_indices = random.sample(range(len(test_dataset)), 5)
    
    for i, idx in enumerate(sample_indices):
        item = test_dataset[idx]
        latin_text = item['latin_text']
        target = item['devanagari_text']
        
        # Translate
        prediction, attentions = translate_sentence(latin_text, latin_vocab, devanagari_vocab, model, device)
        
        # Plot connectivity
        plot_connectivity(latin_text, prediction, attentions)
        
        print(f"Example {i+1}:")
        print(f"  Latin: {latin_text}")
        print(f"  Predicted: {prediction}")
        print(f"  Target: {target}")
        print()

# %%


# %%
