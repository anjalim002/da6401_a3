# Hindi Transliteration: Vanilla RNN and Attention-Based RNN

## Project Overview

This project implements two sequence-to-sequence (Seq2Seq) models for transliterating text from Latin script to Devanagari script for Hindi words. The models are implemented in two Jupyter notebooks:

1. **Vanilla-rnn.ipynb**: A basic Seq2Seq model using Recurrent Neural Networks (RNNs), with options for RNN, GRU, or LSTM cells.
2. **Attention-rnn.ipynb**: An enhanced Seq2Seq model incorporating an attention mechanism to improve performance on longer sequences and capture dependencies between input and output characters.

The models are trained on a Hindi transliteration dataset and use Weights & Biases (W&B) for experiment tracking. The attention-based model includes visualizations of attention maps and connectivity plots to illustrate how the model focuses on input characters during decoding. The project compares the performance of both models, highlighting the improvements achieved by the attention mechanism.

## Prerequisites

To run the notebooks, ensure you have the following dependencies installed:
- Python 3.11
- PyTorch
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Weights & Biases (`wandb`)
- TQDM

You can install the dependencies using:
```bash
pip install torch pandas numpy matplotlib seaborn wandb tqdm
```

### Dataset
The project uses the Hindi transliteration dataset, expected to be located at:
- `/kaggle/input/hindi-dl/hi.translit.sampled.train.tsv`
- `/kaggle/input/hindi-dl/hi.translit.sampled.dev.tsv`
- `/kaggle/input/hindi-dl/hi.translit.sampled.test.tsv`

Each TSV file contains pairs of Latin and Devanagari script words for training, validation, and testing.

### W&B Setup
A Weights & Biases account is required for experiment tracking. Replace the placeholder API key in both notebooks with your own:
```python
wandb.login(key="your_wandb_api_key")
```

### Hardware
A GPU (e.g., NVIDIA Tesla T4) is recommended for faster training. The code automatically falls back to CPU if a GPU is unavailable.

## Running the Notebooks

To achieve the best results, run `Vanilla-rnn.ipynb` first to generate predictions for the vanilla RNN model, followed by `Attention-rnn.ipynb` to generate predictions for the attention-based model and compare their performance.

### Option 1: Run with Default Configuration

#### Vanilla RNN (`Vanilla-rnn.ipynb`)
1. Open `Vanilla-rnn.ipynb` in a Jupyter environment (e.g., Kaggle, Google Colab, or local Jupyter server).
2. Ensure all dependencies are installed.
3. Replace the W&B API key in the first cell with your own.
4. Run all cells. The default configuration in `run_question4()` uses:
   - Embedding dimension: 128
   - Hidden dimension: 256
   - Encoder/Decoder layers: 3
   - Cell type: GRU
   - Dropout: 0.4
   - Batch size: 64
   - Learning rate: 0.001
   - Teacher forcing: 0.7
   - Patience: 3
   - Epochs: 15
5. **Outputs**:
   - Predictions saved to `predictions_vanilla/test_predictions.csv`.
   - Sample predictions and error analysis (e.g., average sequence lengths for correct/incorrect predictions, common error patterns).
   - Results logged to W&B project `transliteration-seq2seq`.

#### Attention RNN (`Attention-rnn.ipynb`)
1. Open `Attention-rnn.ipynb` after running `Vanilla-rnn.ipynb`.
2. Ensure all dependencies are installed.
3. Replace the W&B API key in the first cell with your own.
4. Run all cells. The default configuration in `main()` (with `run_sweep = False`) uses:
   - Embedding dimension: 256
   - Hidden dimension: 256
   - Encoder/Decoder layers: 2
   - Cell type: LSTM
   - Dropout: 0.3
   - Learning rate: 0.001
   - Teacher forcing: 0.7
   - Patience: 3
   - Epochs: 15
5. **Outputs**:
   - Predictions saved to `predictions_attention.csv`.
   - Attention maps saved as `attention_maps.png`.
   - Connectivity visualizations for 5 examples saved as `predictions_attention/connectivity_visualization.png`.
   - Performance comparison with the vanilla model (requires `predictions_vanilla/test_predictions.csv`).
   - Results logged to W&B project `hindi-transliteration-attention`.

### Option 2: Run Hyperparameter Sweep with W&B

#### Vanilla RNN (`Vanilla-rnn.ipynb`)
1. Open `Vanilla-rnn.ipynb`.
2. Replace the W&B API key in the first cell.
3. In the last cell, uncomment the line to run the full sweep in `main()`:
   ```python
   main()
   ```
   Ensure the alternative option is commented out.

4. Run all cells. The sweep tests combinations of:
   - Embedding dimension: [64, 128, 256]
   - Hidden dimension: [128, 256]
   - Encoder/Decoder layers: [1, 2, 3]
   - Cell type: [LSTM, GRU, RNN]
   - Dropout: [0.2, 0.3, 0.4]
   - Batch size: [32, 64, 128]
   - Learning rate: [0.001, 0.0005, 0.0001]
   - Teacher forcing: [0.5, 0.7, 0.9]
   - Epochs: [10, 15]
5. **Outputs**: Same as Option 1, with the best model selected based on validation sequence accuracy.

#### Attention RNN (`Attention-rnn.ipynb`)
1. Open `Attention-rnn.ipynb` after running `Vanilla-rnn.ipynb`.
2. Replace the W&B API key in the first cell.
3. In the last cell, set `run_sweep = True` in `main()`:
   ```python
   run_sweep = True
   ```
4. Run all cells. The sweep uses the same hyperparameter ranges as the vanilla model.
5. **Outputs**: Same as Option 1, with the best model selected based on validation sequence accuracy.

## Notes
- **Execution Order**: Run `Vanilla-rnn.ipynb` first, as `Attention-rnn.ipynb` relies on its predictions for comparison.
- **Performance**: The attention-based model typically outperforms the vanilla model, as it better captures input-output character dependencies, visualized in attention maps and connectivity plots.

## Troubleshooting
- **Missing Dataset**: Ensure the TSV files are in the correct directory (`/kaggle/input/hindi-dl/`).
- **W&B Issues**: Verify your API key and internet connection for logging.
- **GPU Availability**: If training is slow, check that a GPU is available and PyTorch is configured to use it.
