# %%
# Import necessary libraries
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from collections import Counter

import wandb
wandb.login(key="5377a200ae8c04c015415319969d0f2ea19c027c")

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data loading and preprocessing
# def load_data(file_path):
#     """Load data from TSV file"""
#     df = pd.read_csv(file_path, sep='\t', header=None)
#     df.columns = ['latin', 'native']
#     return df
# Data loading and preprocessing
def load_data(file_path):
    """Load data from TSV file"""
    df = pd.read_csv(file_path, sep='\t', header=None, usecols=[0, 1])
    df.columns = ['native', 'latin']
    return df


# Load data files
train_data = load_data('/kaggle/input/hindi-dl/hi.translit.sampled.train.tsv')
val_data = load_data('/kaggle/input/hindi-dl/hi.translit.sampled.dev.tsv')
test_data = load_data('/kaggle/input/hindi-dl/hi.translit.sampled.test.tsv')

print(f"Train data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")
print(f"Test data size: {len(test_data)}")

# Display first few examples
print("\nFirst 5 examples from training data:")
print(train_data.head())

# %%
# Create vocabulary and tokenizer classes
class Vocabulary:
    def __init__(self, freq_threshold=0):
        # Special tokens
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold
        self.idx = 4  # Starting index for new tokens
        
    def __len__(self):
        return len(self.itos)
    
    def build_vocabulary(self, text_list):
        """Build vocabulary from list of texts"""
        frequencies = Counter()
        for text in text_list:
            for char in text:
                frequencies[char] += 1
        
        # Add tokens that meet frequency threshold
        for char, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[char] = self.idx
                self.itos[self.idx] = char
                self.idx += 1
    
    def numericalize(self, text):
        """Convert text to list of indices"""
        numericalized = []
        for char in text:
            if char in self.stoi:
                numericalized.append(self.stoi[char])
            else:
                numericalized.append(self.stoi["<UNK>"])
        return numericalized
    
    def decode(self, indices):
        """Convert indices to text"""
        return ''.join([self.itos[idx] for idx in indices if idx not in [0, 1, 2, 3]])
    
# Create source and target vocabularies
src_vocab = Vocabulary()
tgt_vocab = Vocabulary()

train_data = train_data.dropna()
# Build vocabularies
src_vocab.build_vocabulary(train_data['latin'].tolist())
tgt_vocab.build_vocabulary(train_data['native'].tolist())

print(f"Source vocabulary size: {len(src_vocab)}")
print(f"Target vocabulary size: {len(tgt_vocab)}")

# Create dataset class
class TransliterationDataset(Dataset):
    def __init__(self, df, src_vocab, tgt_vocab):
        self.df = df
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        src_text = self.df.iloc[index]['latin']
        tgt_text = self.df.iloc[index]['native']
        
        # Add SOS and EOS tokens to target
        src_numericalized = [self.src_vocab.stoi["<SOS>"]] + \
                         self.src_vocab.numericalize(src_text) + \
                         [self.src_vocab.stoi["<EOS>"]]
        
        tgt_numericalized = [self.tgt_vocab.stoi["<SOS>"]] + \
                          self.tgt_vocab.numericalize(tgt_text) + \
                          [self.tgt_vocab.stoi["<EOS>"]]
        
        return {
            "src_text": src_text,
            "tgt_text": tgt_text,
            "src": torch.tensor(src_numericalized),
            "tgt": torch.tensor(tgt_numericalized)
        }

# Create padding collate function for batching
def pad_collate(batch):
    src_lens = [len(item["src"]) for item in batch]
    tgt_lens = [len(item["tgt"]) for item in batch]
    
    max_src_len = max(src_lens)
    max_tgt_len = max(tgt_lens)
    
    padded_src = torch.zeros(len(batch), max_src_len).long()
    padded_tgt = torch.zeros(len(batch), max_tgt_len).long()
    
    for i, item in enumerate(batch):
        src_len = len(item["src"])
        tgt_len = len(item["tgt"])
        
        padded_src[i, :src_len] = item["src"]
        padded_tgt[i, :tgt_len] = item["tgt"]
    
    src_texts = [item["src_text"] for item in batch]
    tgt_texts = [item["tgt_text"] for item in batch]
    
    return {
        "src_texts": src_texts,
        "tgt_texts": tgt_texts,
        "src": padded_src,
        "tgt": padded_tgt,
        "src_lens": torch.tensor(src_lens),
        "tgt_lens": torch.tensor(tgt_lens)
    }

# Create datasets
train_dataset = TransliterationDataset(train_data, src_vocab, tgt_vocab)
val_dataset = TransliterationDataset(val_data, src_vocab, tgt_vocab)
test_dataset = TransliterationDataset(test_data, src_vocab, tgt_vocab)

# Test the dataset
sample = train_dataset[0]
print(f"\nSample from dataset:")
print(f"Source text: {sample['src_text']}")
print(f"Target text: {sample['tgt_text']}")
print(f"Source indices: {sample['src']}")
print(f"Target indices: {sample['tgt']}")

# %%
# Encoder model
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, cell_type):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layer (RNN, GRU, or LSTM)
        if cell_type == 'RNN':
            self.rnn = nn.RNN(
                embedding_dim, 
                hidden_dim, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(
                embedding_dim, 
                hidden_dim, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(
                embedding_dim, 
                hidden_dim, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError("cell_type must be one of ['RNN', 'GRU', 'LSTM']")
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, seq_len]
        
        # Pass through embedding
        embedded = self.dropout(self.embedding(x))
        # embedded: [batch_size, seq_len, embedding_dim]
        
        # Pass through RNN
        if self.cell_type == 'LSTM':
            outputs, (hidden, cell) = self.rnn(embedded)
            # hidden: [num_layers, batch_size, hidden_dim]
            # cell: [num_layers, batch_size, hidden_dim]
            return outputs, (hidden, cell)
        else:  # RNN or GRU
            outputs, hidden = self.rnn(embedded)
            # hidden: [num_layers, batch_size, hidden_dim]
            return outputs, hidden

# Decoder model
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, cell_type):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layer (RNN, GRU, or LSTM)
        if cell_type == 'RNN':
            self.rnn = nn.RNN(
                embedding_dim, 
                hidden_dim, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(
                embedding_dim, 
                hidden_dim, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(
                embedding_dim, 
                hidden_dim, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError("cell_type must be one of ['RNN', 'GRU', 'LSTM']")
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden):
        # x: [batch_size, 1]
        # hidden: [num_layers, batch_size, hidden_dim] for RNN/GRU
        # or tuple of ([num_layers, batch_size, hidden_dim], [num_layers, batch_size, hidden_dim]) for LSTM
        
        # Pass through embedding
        embedded = self.dropout(self.embedding(x))
        # embedded: [batch_size, 1, embedding_dim]
        
        # Pass through RNN
        if self.cell_type == 'LSTM':
            output, (hidden, cell) = self.rnn(embedded, hidden)
            # output: [batch_size, 1, hidden_dim]
            # hidden: [num_layers, batch_size, hidden_dim]
            # cell: [num_layers, batch_size, hidden_dim]
        else:  # RNN or GRU
            output, hidden = self.rnn(embedded, hidden)
            # output: [batch_size, 1, hidden_dim]
            # hidden: [num_layers, batch_size, hidden_dim]
        
        # Pass through output layer
        prediction = self.fc_out(output.squeeze(1))
        # prediction: [batch_size, vocab_size]
        
        if self.cell_type == 'LSTM':
            return prediction, (hidden, cell)
        else:
            return prediction, hidden

# Seq2Seq model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # Ensure encoder and decoder have same dimensions and cell types
        assert encoder.hidden_dim == decoder.hidden_dim
        assert encoder.num_layers == decoder.num_layers
        assert encoder.cell_type == decoder.cell_type
        
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len]
        
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.vocab_size
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        
        # Encode the source sequence
        if self.encoder.cell_type == 'LSTM':
            _, (hidden, cell) = self.encoder(src)
            # hidden: [num_layers, batch_size, hidden_dim]
            # cell: [num_layers, batch_size, hidden_dim]
        else:  # RNN or GRU
            _, hidden = self.encoder(src)
            # hidden: [num_layers, batch_size, hidden_dim]
        
        # First input to the decoder is the <SOS> token
        input = tgt[:, 0].unsqueeze(1)  # [batch_size, 1]
        
        for t in range(1, tgt_len):
            # Pass through decoder
            if self.decoder.cell_type == 'LSTM':
                output, (hidden, cell) = self.decoder(input, (hidden, cell))
                # output: [batch_size, vocab_size]
                # hidden: [num_layers, batch_size, hidden_dim]
                # cell: [num_layers, batch_size, hidden_dim]
            else:  # RNN or GRU
                output, hidden = self.decoder(input, hidden)
                # output: [batch_size, vocab_size]
                # hidden: [num_layers, batch_size, hidden_dim]
            
            # Store output
            outputs[:, t, :] = output
            
            # Decide whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Get the highest predicted token
            top1 = output.argmax(1)
            
            # Use ground truth or predicted token as next input
            input = tgt[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
            
        return outputs
    
    def translate(self, src, max_len=100):
        # src: [batch_size, src_len]
        
        batch_size = src.shape[0]
        
        # Encode the source sequence
        if self.encoder.cell_type == 'LSTM':
            _, (hidden, cell) = self.encoder(src)
        else:  # RNN or GRU
            _, hidden = self.encoder(src)
        
        # First input to the decoder is the <SOS> token
        input = torch.tensor([self.decoder.embedding.num_embeddings - 3] * batch_size).unsqueeze(1).to(self.device)  # SOS token
        
        outputs = []
        
        for t in range(max_len):
            # Pass through decoder
            if self.decoder.cell_type == 'LSTM':
                output, (hidden, cell) = self.decoder(input, (hidden, cell))
            else:  # RNN or GRU
                output, hidden = self.decoder(input, hidden)
            
            # Get the highest predicted token
            top1 = output.argmax(1)
            
            # Add to outputs
            outputs.append(top1.unsqueeze(1))
            
            # Use predicted token as next input
            input = top1.unsqueeze(1)
            
            # Break if EOS token is predicted (assuming EOS is 2)
            if all(top1 == 2):
                break
                
        # Concatenate all predictions
        outputs = torch.cat(outputs, dim=1)
        
        return outputs

# Initialize the model
def create_model(config, src_vocab_size, tgt_vocab_size, device):
    encoder = Encoder(
        vocab_size=src_vocab_size,
        embedding_dim=config['emb_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['enc_layers'],
        dropout=config['dropout'],
        cell_type=config['cell_type']
    )
    
    decoder = Decoder(
        vocab_size=tgt_vocab_size,
        embedding_dim=config['emb_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['dec_layers'],
        dropout=config['dropout'],
        cell_type=config['cell_type']
    )
    
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    return model

# Calculate model parameters and computations
def calculate_model_stats(m, k, V, T):
    """
    Calculate number of parameters and computations for a simple Seq2Seq model
    
    Args:
        m: embedding size
        k: hidden state size
        V: vocabulary size
        T: sequence length
        
    Returns:
        num_params: number of parameters
        num_comps: number of computations
    """
    # Embedding parameters
    emb_params = V * m
    
    # RNN cell parameters (assuming vanilla RNN)
    # Input matrix: m x k, hidden matrix: k x k, bias: k
    rnn_params = (m * k) + (k * k) + k
    
    # Output layer parameters
    output_params = k * V + V
    
    # Total parameters for encoder and decoder
    num_params = 2 * emb_params + 2 * rnn_params + output_params
    
    # Total computations
    # For each time step:
    # - embedding lookup: m
    # - RNN computation: m*k + k*k
    # - output layer: k*V
    per_step_comp = m + (m * k) + (k * k) + (k * V)
    num_comps = T * per_step_comp  # For encoder + decoder
    
    return num_params, num_comps

# Example: m=64, k=128, V=100, T=10
m, k, V, T = 64, 128, 100, 10
num_params, num_comps = calculate_model_stats(m, k, V, T)
print(f"Number of parameters: {num_params}")
print(f"Number of computations: {num_comps}")

# %%
# Initialize training and validation functions
def train_epoch(model, iterator, optimizer, criterion, teacher_forcing_ratio, clip, device):
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(iterator, desc="Training"):
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, tgt, teacher_forcing_ratio)
        
        # Calculate loss (ignore pad tokens)
        output_dim = output.shape[-1]
        
        # Reshape for cross entropy
        output = output[:, 1:].reshape(-1, output_dim)
        tgt = tgt[:, 1:].reshape(-1)
        
        loss = criterion(output, tgt)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Update parameters
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    correct_predictions = 0
    total_samples = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating"):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            src_texts = batch["src_texts"]
            tgt_texts = batch["tgt_texts"]
            
            # Forward pass (no teacher forcing)
            output = model(src, tgt, 0)
            
            # Get predictions (translate mode)
            predictions = model.translate(src)
            
            # Calculate loss (ignore pad tokens)
            output_dim = output.shape[-1]
            
            # Reshape for cross entropy
            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, tgt)
            
            epoch_loss += loss.item()
            
            # Calculate accuracy
            for i in range(len(src_texts)):
                pred_seq = predictions[i].cpu().numpy()
                # Remove padding, SOS, EOS
                pred_seq = [idx for idx in pred_seq if idx not in [0, 1, 2]]
                pred_text = ''.join([tgt_vocab.itos[idx] for idx in pred_seq])
                
                # Compare with target text (ignoring SOS, EOS)
                target_text = tgt_texts[i]
                
                all_predictions.append(pred_text)
                all_targets.append(target_text)
                
                # Check if prediction matches target
                if pred_text == target_text:
                    correct_predictions += 1
                
                total_samples += 1
    
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    
    return epoch_loss / len(iterator), accuracy, all_predictions, all_targets

# Training loop
def train_model(config, model, train_loader, val_loader, criterion, optimizer, scheduler, device, run):
    # Set parameters
    n_epochs = config["epochs"]
    clip = 1.0
    teacher_forcing_ratio = config["teacher_forcing"]
    patience = config["patience"]
    
    # Initialize for early stopping
    best_valid_loss = float('inf')
    counter = 0
    best_model = None
    
    for epoch in range(n_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, teacher_forcing_ratio, clip, device)
        
        # Evaluate
        valid_loss, valid_accuracy, _, _ = evaluate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(valid_loss)
        
        # Log to W&B
        run.log({
            "train_loss": train_loss,
            "val_loss": valid_loss,
            "val_sequence_accuracy": valid_accuracy,
            "epoch": epoch + 1
        })
        
        print(f"Epoch: {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {valid_loss:.4f}")
        print(f"Validation Accuracy: {valid_accuracy:.4f}")
        
        # Early stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Load best model
    model.load_state_dict(best_model)
    
    return model

# Function to test the model and get sample predictions
def test_model(model, test_loader, device, tgt_vocab):
    model.eval()
    all_predictions = []
    all_targets = []
    all_sources = []
    
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            src_texts = batch["src_texts"]
            tgt_texts = batch["tgt_texts"]
            
            # Get predictions
            predictions = model.translate(src)
            
            # Process predictions
            for i in range(len(src_texts)):
                pred_seq = predictions[i].cpu().numpy()
                # Remove padding, SOS, EOS
                pred_seq = [idx for idx in pred_seq if idx not in [0, 1, 2]]
                pred_text = ''.join([tgt_vocab.itos[idx] for idx in pred_seq])
                
                # Store results
                all_sources.append(src_texts[i])
                all_predictions.append(pred_text)
                all_targets.append(tgt_texts[i])
                
                # Check if prediction matches target
                if pred_text == tgt_texts[i]:
                    correct_predictions += 1
                
                total_samples += 1
    
    # Calculate accuracy
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    
    # Sample some predictions for display
    num_samples = min(10, len(all_sources))
    sample_indices = np.random.choice(len(all_sources), num_samples, replace=False)
    
    sample_data = {
        "Source": [all_sources[i] for i in sample_indices],
        "Target": [all_targets[i] for i in sample_indices],
        "Prediction": [all_predictions[i] for i in sample_indices],
        "Correct": [all_predictions[i] == all_targets[i] for i in sample_indices]
    }
    
    return accuracy, all_sources, all_targets, all_predictions, sample_data

# Save predictions to file
def save_predictions(sources, targets, predictions, filename):
    df = pd.DataFrame({
        "Source": sources,
        "Target": targets,
        "Prediction": predictions,
        "Correct": [p == t for p, t in zip(predictions, targets)]
    })
    df.to_csv(filename, index=False)
    return df

# %%
# Initialize W&B sweep
def train_with_config():
    # Login to wandb
    wandb.login()
    
    # Define sweep config
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
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="transliteration-seq2seq")
    
    

# %%
# Initialize W&B sweep
def train_with_config():
    # Login to wandb
    wandb.login()
    
    # Define sweep config
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
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="transliteration-seq2seq")
    
    # Define the training function for each sweep run
    def train_sweep():
        # Initialize wandb run
        run = wandb.init()
        
        # Get hyperparameters for this run
        config = wandb.config
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=pad_collate
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=pad_collate
        )
        
        # Create model
        model = create_model(
            config,
            len(src_vocab),
            len(tgt_vocab),
            device
        )
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1, verbose=True
        )
        
        # Train the model
        model = train_model(
            config,
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            run
        )
        
        # Test on validation set for final metrics
        valid_loss, valid_accuracy, valid_preds, valid_targets = evaluate(
            model, val_loader, criterion, device
        )
        
        # Log final metrics
        run.log({
            "final_val_loss": valid_loss,
            "final_val_accuracy": valid_accuracy
        })
        
        # Save the model
        torch.save(model.state_dict(), f"model_{run.id}.pt")
        
        # Finish the run
        run.finish()
    
    # Run the sweep
    wandb.agent(sweep_id, train_sweep, count=10)  # Run 10 experiments
    
    return sweep_id

# %%
# Main execution script
def main():
    # Run sweep to find best hyperparameters
    print("Starting hyperparameter sweep...")
    sweep_id = train_with_config()
    
    # Get the best run from the sweep
    api = wandb.Api()
    sweep = api.sweep(f"da24m002-indian-institute-of-technology-madras/transliteration-seq2seq/{sweep_id}")
    
    # Sort runs by validation accuracy
    runs = sorted(sweep.runs, key=lambda run: run.summary.get('val_sequence_accuracy', 0), reverse=True)
    
    if runs:
        best_run = runs[0]
        best_config = {
            k: v for k, v in best_run.config.items() 
            if k in ['emb_dim', 'hidden_dim', 'enc_layers', 'dec_layers', 
                    'cell_type', 'dropout', 'batch_size', 'learning_rate', 
                    'teacher_forcing', 'patience', 'epochs']
        }
        
        print(f"Best run: {best_run.name}")
        print(f"Best validation accuracy: {best_run.summary.get('val_sequence_accuracy', 0)}")
        print(f"Best configuration: {best_config}")
        
        # Create test dataloader
        test_loader = DataLoader(
            test_dataset,
            batch_size=best_config['batch_size'],
            shuffle=False,
            collate_fn=pad_collate
        )
        
        # Create model with best config
        best_model = create_model(
            best_config,
            len(src_vocab),
            len(tgt_vocab),
            device
        )
        
        # Load best model weights
        best_model.load_state_dict(torch.load(f"model_{best_run.id}.pt"))
        
        # Evaluate on test set
        print("Evaluating best model on test set...")
        test_accuracy, test_sources, test_targets, test_preds, sample_data = test_model(
            best_model, test_loader, device, tgt_vocab
        )
        
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # Create directory for predictions if it doesn't exist
        if not os.path.exists('predictions_vanilla'):
            os.makedirs('predictions_vanilla')
        
        # Save predictions
        save_predictions(
            test_sources,
            test_targets,
            test_preds,
            'predictions_vanilla/test_predictions.csv'
        )
        
        # Print sample predictions
        print("\nSample predictions:")
        for i in range(len(sample_data['Source'])):
            print(f"Source: {sample_data['Source'][i]}")
            print(f"Target: {sample_data['Target'][i]}")
            print(f"Prediction: {sample_data['Prediction'][i]}")
            print(f"Correct: {'✓' if sample_data['Correct'][i] else '✗'}")
            print("---")
    else:
        print("No runs found in sweep.")

# Run the code for question 4 directly (testing on test data)
def run_question4(best_config=None):
    """
    Run evaluation on the test set directly using a predefined configuration
    This can be used if you want to skip the sweep
    """
    if best_config is None:
        # Define a default configuration if none provided
        best_config = {
            'emb_dim': 128,
            'hidden_dim': 256,
            'enc_layers': 3,
            'dec_layers': 3,
            'cell_type': 'GRU',
            'dropout': 0.4,
            'batch_size': 64,
            'learning_rate': 0.001,
            'teacher_forcing': 0.7,
            'patience': 3,
            'epochs': 15
        }
    
    # Initialize wandb
    wandb.init(project="transliteration-direct-test", config=best_config)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=best_config['batch_size'],
        shuffle=True,
        collate_fn=pad_collate
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=best_config['batch_size'],
        shuffle=False,
        collate_fn=pad_collate
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=best_config['batch_size'],
        shuffle=False,
        collate_fn=pad_collate
    )
    
    # Create model
    model = create_model(
        best_config,
        len(src_vocab),
        len(tgt_vocab),
        device
    )
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    optimizer = optim.Adam(model.parameters(), lr=best_config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, verbose=True
    )
    
    # Train the model
    print("Training model with best configuration...")
    model = train_model(
        best_config,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        wandb.run
    )
    
    # Save the trained model
    torch.save(model.state_dict(), f"best_model.pt")
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    test_accuracy, test_sources, test_targets, test_preds, sample_data = test_model(
        model, test_loader, device, tgt_vocab
    )
    
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Create directory for predictions if it doesn't exist
    if not os.path.exists('predictions_vanilla'):
        os.makedirs('predictions_vanilla')
    
    # Save predictions
    predictions_df = save_predictions(
        test_sources,
        test_targets,
        test_preds,
        'predictions_vanilla/test_predictions.csv'
    )
    
    # Log predictions table to wandb
    wandb.log({
        "test_accuracy": test_accuracy,
        "predictions_table": wandb.Table(dataframe=predictions_df.sample(min(100, len(predictions_df))))
    })
    
    # Print sample predictions
    print("\nSample predictions:")
    sample_indices = np.random.choice(len(predictions_df), min(10, len(predictions_df)), replace=False)
    for idx in sample_indices:
        print(f"Source: {predictions_df.iloc[idx]['Source']}")
        print(f"Target: {predictions_df.iloc[idx]['Target']}")
        print(f"Prediction: {predictions_df.iloc[idx]['Prediction']}")
        print(f"Correct: {'✓' if predictions_df.iloc[idx]['Correct'] else '✗'}")
        print("---")
    
    # Analyze errors
    error_indices = predictions_df[~predictions_df['Correct']].index
    error_df = predictions_df.loc[error_indices]
    
    print(f"\nTotal errors: {len(error_df)}")
    print(f"Error rate: {len(error_df) / len(predictions_df):.4f}")
    
    # Simple error analysis
    if len(error_df) > 0:
        error_lengths = error_df['Source'].str.len()
        correct_lengths = predictions_df[predictions_df['Correct']]['Source'].str.len()
        
        print(f"Average length of error cases: {error_lengths.mean():.2f}")
        print(f"Average length of correct cases: {correct_lengths.mean():.2f}")
        
        # Find most common error patterns
        error_pairs = list(zip(error_df['Target'], error_df['Prediction']))
        common_errors = Counter(error_pairs).most_common(5)
        
        print("\nMost common errors (Target -> Prediction):")
        for (target, pred), count in common_errors:
            print(f"  '{target}' -> '{pred}': {count} occurrences")
    
    wandb.finish()
    
    return model, test_accuracy, predictions_df

# You can choose to run either the full sweep or just Question 4 directly
if __name__ == "__main__":
    # Choose one of the following:
    
    # Option 1: Run the full sweep and evaluation
    # main()
    
    # Option 2: Run Question 4 directly with a predefined config
    run_question4()

# %%
