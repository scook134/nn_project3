# ORAL_EXAM_PREP

This file is an oral-exam survival guide for the actual repository. It is intentionally strict: if a claim is not supported by the notebooks or `report.md`, it is either removed or labeled as uncertain.

The repo has four top-level files:

- `AR_RNN-v8.ipynb`
- `STUDENT_discrete_time_series_llm_notebook.ipynb`
- `ETTh1.csv`
- `report.md`

The project is best explained as a comparison between two formulations of the same forecasting task:

1. `AR_RNN-v8.ipynb`: continuous probabilistic forecasting with Gaussian outputs
2. `STUDENT_discrete_time_series_llm_notebook.ipynb`: discrete/tokenized forecasting with next-token prediction

# 1. Project Overview

## What problem this project solves

The project studies **probabilistic univariate time-series forecasting** on the `OT` signal from `ETTh1.csv`.

The central comparison is:

- predict the next value as a **Gaussian distribution** over continuous values
- or predict the next value as a **categorical distribution over 128 bins**

So the real project question is:

- can the forecasting problem be reformulated from continuous-value prediction into token prediction, while still giving reasonable or strong forecasting performance?

## What dataset/task it uses

Both notebooks explicitly do:

- `df = pd.read_csv("ETTh1.csv")`
- `series = df["OT"].values.astype(np.float32)`

So the task is:

- **univariate** forecasting
- target variable: `OT`

Both notebooks then:

- standardize the full `OT` series
- subsample by `10`
- create many random shifted windows

Saved outputs show:

- subsampled length: `1742`
- `T = 200`
- `N = 5000`
- train/test split: `4000 / 1000`
- `T_train = 100`

## What the input and output are

### Continuous notebook: `AR_RNN-v8.ipynb`

Input:

- continuous standardized values
- shaped as `(N, T, 1)` during model forward passes

Output from `RNN.forward(...)`, `LSTM.forward(...)`, and `LSTMTransformer.forward(...)`:

- `mean`: `(N, T, 1)`
- `std`: `(N, T, 1)`
- sampled output: `(N, T, 1)`

### Discrete notebook: `STUDENT_discrete_time_series_llm_notebook.ipynb`

Input:

- token IDs in `[0, 127]`
- shaped as `(B, T)`

Output from `DiscreteRNN.forward(...)`, `DiscreteLSTM.forward(...)`, `DiscreteTransformer.forward(...)`, and `DiscreteLSTMTransformer.forward(...)`:

- logits over bins
- shaped as `(B, T, 128)`

## What the final model is trying to learn

Because the windows are built as:

- `X = seq[:-1]`
- `Y = seq[1:]`

every model is learning a one-step-ahead conditional prediction problem at each time step.

More precisely:

- continuous models learn `p(x_{t+1} | x_1, ..., x_t)` under a Gaussian parameterization
- discrete models learn `p(bin_{t+1} | bin_1, ..., bin_t)` under a categorical parameterization

## What the main experimental goal is

The repo tries to compare:

- simple recurrent baselines (`RNN`, `LSTM`)
- against hybrid recurrent-attention models (`LSTMTransformer`, `DiscreteLSTMTransformer`)
- and, more importantly, continuous-output forecasting versus token-output forecasting

The safest claim is:

- the repo demonstrates that the tokenized formulation is workable and can perform strongly in the saved experiment

The unsafe claim is:

- “discrete forecasting is universally better”

That claim is **not** supported by this repo.

# 2. Repository Map

## `ETTh1.csv`

What it does:

- raw data file

Why it exists:

- both notebooks load it and extract `OT`

How it connects:

- it is the only external data source used by the code

## `AR_RNN-v8.ipynb`

What it does:

- implements the continuous probabilistic forecasting pipeline

Key classes/functions:

- `RNN`
- `RNN_extended`
- `LSTM`
- `LSTM_extended`
- `PositionalEncoding`
- `LSTMTransformer`
- `TR_extended`

Why it exists:

- it is the baseline notebook using Gaussian outputs

How it connects:

- it provides the original setup that the discrete notebook conceptually extends

Important caveat:

- the final “Transformer” result is not a pure Transformer
- in code it is `LSTMTransformer`, a hybrid LSTM + Transformer encoder model

## `STUDENT_discrete_time_series_llm_notebook.ipynb`

What it does:

- implements the discrete/tokenized forecasting pipeline

Key classes/functions:

- `quantize_uniform(...)`
- `dequantize_uniform(...)`
- `TokenSequenceDataset`
- `DiscreteRNN`
- `DiscreteLSTM`
- `DiscreteTransformer`
- `DiscreteLSTMTransformer`
- `token_cross_entropy_loss(...)`
- `sample_tokens(...)`
- `tokens_to_values(...)`
- `train_discrete_model(...)`
- `evaluate_one_step(...)`
- `evaluate_autoregressive(...)`

Why it exists:

- it reframes forecasting as next-token prediction

How it connects:

- it uses the same `OT` series and broadly the same window structure as the continuous notebook

Important caveat:

- `DiscreteTransformer` is defined but not included in the final trained `models` dictionary

## `report.md`

What it does:

- summarizes the project and lists saved results

Why it exists:

- it is the written narrative layer of the repo

How it connects:

- useful for confirming notebook outputs, but should not replace the notebooks when answering code-level questions

# 3. End-to-End Pipeline

## Step 1: Load and standardize the signal

In both notebooks:

- `df = pd.read_csv("ETTh1.csv")`
- `series = df["OT"].values.astype(np.float32)`
- `mean = series.mean()`
- `std = series.std()`
- `series = (series - mean) / std`

Strict caveat:

- normalization is fit on the full series before the train/test split, so there is train/test leakage

## Step 2: Subsample

In both notebooks:

- `subsample_factor = 10`
- `series_sub = series[::subsample_factor]`

Saved output:

- `Total length after subsampling: 1742`

Why this likely helps:

- reduces sequence density
- makes long-range structure more prominent

Tradeoff:

- may discard short-range dynamics

## Step 3: Quantize the series in the discrete notebook

Only `STUDENT_discrete_time_series_llm_notebook.ipynb` does this.

Relevant code:

- `NUM_BINS = 128`
- `train_min = series_sub.min()`
- `train_max = series_sub.max()`
- `quantize_uniform(...)`
- `dequantize_uniform(...)`

Quantization logic:

1. normalize into `[0, 1]`
2. multiply by `128`
3. apply `floor`
4. clip into `[0, 127]`

Saved output:

- `Unique bins used: 121`

Strict caveat:

- despite the names `train_min` and `train_max`, those bounds are computed from the full subsampled series, not from a post-split training subset

## Step 4: Build shifted windows

Continuous notebook:

- `T = 200`
- `N = 5000`
- `seq = series_sub[start : start + T + 1]`
- `X_list.append(seq[:-1])`
- `Y_list.append(seq[1:])`

Discrete notebook:

- `T = 200`
- `N = 5000`
- `X = np.stack([series_bins[s:s+T] for s in starts], axis=0)`
- `Y = np.stack([series_bins[s+1:s+T+1] for s in starts], axis=0)`

Saved shapes:

- `X shape: (5000, 200)`
- `Y shape: (5000, 200)`

Meaning:

- each position predicts the next position

## Step 5: Split into train and test

Saved outputs:

- train: `(4000, 200)`
- test: `(1000, 200)`

Difference:

- `AR_RNN-v8.ipynb` uses `np.random.permutation(...)`
- `STUDENT_discrete_time_series_llm_notebook.ipynb` uses `rng = np.random.default_rng(0)`

So the discrete notebook has an explicit saved random seed and the continuous one does not.

## Step 6: Define the context length

Both notebooks define:

- `T_train = 100`

But use it differently.

Supported by code:

- continuous notebook training actually uses `X_train[:, :T_train]` and `Y_train[:, :T_train]`
- discrete notebook forecasting/evaluation uses `T_train = 100`, but training still consumes full length-200 sequences from `train_loader`

This is a likely exam question.

## Step 7: Train the models

### Continuous notebook

`RNN_extended.trainloop(...)` and `LSTM_extended.trainloop(...)` do:

1. reshape `x` and `y` into `(N_train, T_train, 1)`
2. forward pass through the full training set
3. compute `gaussian_nll(y, mean, std)`
4. `loss.backward()`
5. `nn.utils.clip_grad_norm_(..., max_norm=2.0)`
6. `optim.step()`
7. repeat for `250` iterations

Important precision:

- this is full-batch optimization with Adam
- the sampled output is returned by `forward(...)`, but the final loss uses `mean` and `std`

For the hybrid model:

- `TR_extended` is defined twice
- the second definition overwrites the first
- the second definition uses Gaussian NLL and is the one instantiated by `my_TF = TR_extended(...)`

### Discrete notebook

`train_discrete_model(...)` and `DiscreteRNNExtended.trainloop(...)` do:

1. iterate over minibatches from `DataLoader`
2. compute `logits, _ = model(x_batch)`
3. compute `token_cross_entropy_loss(logits, y_batch)`
4. `loss.backward()`
5. clip gradients to `2.0`
6. update with Adam

Batch shapes:

- `x_batch`: `(batch_size, 200)`
- `y_batch`: `(batch_size, 200)`
- `logits`: `(batch_size, 200, 128)`

Inside `token_cross_entropy_loss(...)`:

- logits `(B, T, C)` -> `(B*T, C)`
- targets `(B, T)` -> `(B*T,)`

So cross-entropy supervision is applied at every time step.

## Step 8: Evaluate

### Continuous notebook

One-step evaluation:

- `T_test = 100`
- build input `(N_test, 100, 1)`
- compare predicted `mean` to `Y_test[:, :100]`

RNN/LSTM autoregressive evaluation:

1. take the last sampled context point
2. keep the final recurrent hidden state
3. feed sampled predictions back in
4. forecast `H_forecasting = 20`

Important nuance:

- the context block uses predicted means
- the forecast block uses sampled values

Hybrid `LSTMTransformer` evaluation is different:

1. keep a full context window tensor
2. run the whole window through the model
3. take the sampled last output
4. append it and drop the oldest step

So the hybrid uses sliding-window autoregression, not hidden-state recurrence.

### Discrete notebook

One-step evaluation in `evaluate_one_step(...)`:

1. compute logits
2. compute cross-entropy
3. compute `softmax`
4. compute expected bin index using `sum(prob * bin_positions)`
5. dequantize expected bins
6. compute MSE in value space

Important nuance:

- this is expected-bin decoding, not argmax decoding

Autoregressive evaluation in `evaluate_autoregressive(...)`:

1. seed with `X_test[i, :context_len]`
2. call `sample_tokens(...)`
3. sample each next token with `torch.multinomial(...)`
4. dequantize future bins
5. compute forecast MSE

Very important caveat:

- the saved autoregressive metric uses `num_sequences=64`
- not the full 1000-sequence test set

## Step 9: Save/load behavior

There are no external checkpoint files in the repo.

Results are stored mainly as:

- executed notebook outputs
- plots inside notebooks
- summary text in `report.md`

# 4. Model Architecture

## Continuous `RNN`

Defined in:

- `AR_RNN-v8.ipynb`, class `RNN`

Forward flow:

1. input `x`: `(N, T, 1)`
2. `self.rnn(x)` -> `r_out`: `(N, T, hidden_dim)`, `hidden`: `(n_layers, N, hidden_dim)`
3. reshape `r_out` to `(N*T, hidden_dim)`
4. `fc0 -> ReLU -> fc1` -> `(N*T, 2)`
5. split:
   - `media`: `(N*T,)`
   - `std`: `(N*T,)`
6. `std = softplus(raw_std)`
7. sample `media + std * noise`
8. reshape `media`, `std`, `sample` to `(N, T, 1)`

Instantiated trained version:

- `my_rnn = RNN_extended(... hidden_dim=32, n_layers=1, sigma=0.5, lr=0.005)`

## Continuous `LSTM`

Defined in:

- class `LSTM`

Difference from `RNN`:

- recurrent core is `nn.LSTM(...)`
- hidden state is a tuple `(h_T, c_T)`

Shapes:

- input: `(N, T, 1)`
- recurrent output: `(N, T, hidden_dim)`
- hidden tuple:
  - `h_T`: `(n_layers, N, hidden_dim)`
  - `c_T`: `(n_layers, N, hidden_dim)`

Instantiated trained version:

- `my_lstm = LSTM_extended(... hidden_dim=32, n_layers=1, sigma=0.5, lr=0.005)`

## Continuous `LSTMTransformer`

Defined in:

- class `LSTMTransformer`

This is the exact architecture in code:

1. input `(N, T, 1)`
2. `nn.LSTM(input_size=1, hidden_size=32, num_layers=1, batch_first=True)`
3. `embedding_proj`: project hidden size -> `d_model`
4. `PositionalEncoding`
5. `nn.TransformerEncoder(...)`
6. flatten to `(N*T, d_model)`
7. `fc0 -> ReLU -> fc1`
8. split into Gaussian `mean` and `std`
9. sample from that Gaussian

Instantiated trained version:

- `my_TF = TR_extended(d_model=64, num_layers=2, nhead=4, dim_feedforward=16, hidden_size=32, ... )`

Important precision:

- the notebook calls this a Transformer result, but the class is a hybrid `LSTMTransformer`

## Discrete `DiscreteRNN`

Defined in:

- `DiscreteRNN`

Forward flow:

1. token IDs `(B, T)`
2. `nn.Embedding(128, 64)` -> `(B, T, 64)`
3. `nn.RNN(input_size=64, hidden_size=32, nonlinearity="relu", batch_first=True)`
4. linear head -> `(B, T, 128)`

Returned values:

- `logits`: `(B, T, 128)`
- `hidden`

## Discrete `DiscreteLSTM`

Defined in:

- `DiscreteLSTM`

Forward flow:

1. token IDs `(B, T)`
2. embedding `(B, T, 64)`
3. `nn.LSTM(...)` -> `(B, T, 32)`
4. linear head -> `(B, T, 128)`

Important precision:

- `drop_prob=0.3` is passed to the constructor
- but because `n_layers=1` in the trained model, effective LSTM dropout is `0.0`

## Discrete `DiscreteTransformer`

Defined in:

- `DiscreteTransformer`

Forward flow:

1. token IDs `(B, T)`
2. embedding `(B, T, 64)`
3. positional encoding
4. causal mask from `causal_mask(seq_len, device)`
5. `nn.TransformerEncoder(...)`
6. linear head -> `(B, T, 128)`

Important caveat:

- defined, but not included in the final trained/evaluated `models` dictionary

## Discrete `DiscreteLSTMTransformer`

Defined in:

- `DiscreteLSTMTransformer`

Forward flow:

1. token IDs `(B, T)`
2. embedding `(B, T, 64)`
3. LSTM -> `(B, T, 32)`
4. projection `32 -> 64`
5. positional encoding
6. causal Transformer encoder
7. linear head -> `(B, T, 128)`

Instantiated trained version:

- `DiscreteLSTMTransformer(num_bins=128, d_model=64, lstm_hidden_dim=32, lstm_layers=1, nhead=4, num_layers=1, dim_feedforward=128, dropout=0.1, max_len=T + 10)`

## Why these architectures make sense

Supported interpretations:

- `RNN` is the simplest autoregressive baseline
- `LSTM` adds gating and longer-memory capacity
- hybrid LSTM+Transformer models try to combine local sequential inductive bias with richer contextual mixing

Unsupported interpretation:

- “The repo proves the Transformer mechanism itself is the reason for all gains”

That is too strong because the hybrid models change several things at once.

# 5. Training Details

## Shared setup

- target signal: `OT`
- subsample factor: `10`
- `T = 200`
- `N = 5000`
- train/test split: `4000 / 1000`
- `T_train = 100`

## Continuous notebook hyperparameters

### `RNN_extended`

- `hidden_dim=32`
- `n_layers=1`
- `lr=0.005`
- `num_iter=250`
- loss: `gaussian_nll(...)`
- gradient clipping: `max_norm=2.0`
- training slice: first 100 steps only
- training style: full-batch

### `LSTM_extended`

- `hidden_dim=32`
- `n_layers=1`
- `sigma=0.5`
- `lr=0.005`
- `num_iter=250`
- loss: `gaussian_nll(...)`
- gradient clipping: `max_norm=2.0`
- training slice: first 100 steps only
- training style: full-batch

### `TR_extended`

- `hidden_size=32`
- `d_model=64`
- `nhead=4`
- `num_layers=2`
- `dim_feedforward=16`
- `lr=0.005`
- `num_iter=250`
- loss: Gaussian NLL in the final executed definition
- gradient clipping: `max_norm=2.0`
- training slice: first 100 steps only
- training style: full-batch

## Discrete notebook hyperparameters

- `NUM_BINS = 128`
- `batch_size = 64`
- `num_epochs = 12`
- `lr = 1e-3`
- gradient clipping: `max_norm = 2.0`

### `DiscreteRNN`

- `d_model=64`
- `hidden_dim=32`
- `n_layers=1`

### `DiscreteLSTM`

- `d_model=64`
- `hidden_dim=32`
- `n_layers=1`

### `DiscreteLSTMTransformer`

- `d_model=64`
- `lstm_hidden_dim=32`
- `lstm_layers=1`
- `nhead=4`
- `num_layers=1`
- `dim_feedforward=128`
- `dropout=0.1`

## Losses and what they imply

### Gaussian NLL

Used in the final baseline training code.

Implication:

- the model is trained to fit both central tendency and uncertainty width

### Token cross-entropy

Used in the discrete notebook.

Implication:

- the model is trained as a categorical predictor at each time step
- uncertainty is represented by a probability distribution over bins

## Metrics

### Continuous notebook

- one-step MSE
- horizon MSE for `H_forecasting = 20`

### Discrete notebook

- one-step cross-entropy
- one-step MSE after dequantization
- autoregressive forecast MSE after dequantization

Strict caveat:

- the discrete autoregressive MSE is computed on 64 sampled test sequences, not the full test set

## Regularization/stabilization actually present

- gradient clipping in both notebooks
- dropout only in some LSTM/Transformer definitions

Not present:

- no weight decay
- no validation split
- no early stopping
- no repeated-seed study

# 6. Architectural Decisions and Tradeoffs

## Gaussian output vs token output

What the repo does:

- compares both

Why it makes sense:

- Gaussian output is standard for continuous forecasting
- token output removes the single-Gaussian output restriction

Tradeoff:

- Gaussian output is smooth and natural for real numbers, but may be too restrictive
- token output is flexible, but introduces quantization error

## RNN vs LSTM

What the repo does:

- uses both as matched recurrent baselines in both formulations

Tradeoff:

- RNN is simpler
- LSTM usually handles longer dependencies better

## Hybrid LSTM + Transformer

What the repo does:

- uses `LSTMTransformer` and `DiscreteLSTMTransformer`

Why it makes sense:

- LSTM gives local sequential processing
- Transformer encoder layers mix information more globally

Tradeoff:

- more capacity and flexibility
- harder attribution: gains may come from several changes, not just “attention”

## One-step training vs long-horizon rollout

What the repo does:

- trains one-step predictors
- tests them autoregressively

Tradeoff:

- good one-step accuracy does not guarantee stable future rollout

## Uniform quantization

What the repo does:

- uses `quantize_uniform(...)` with 128 bins

Tradeoff:

- simple and easy to explain
- may waste bins if the distribution is uneven

## Experimental hygiene

What the repo does not do:

- train-only normalization
- train-only quantization bounds
- validation split
- seed sweeps
- matched horizons across continuous and discrete final comparisons

# 7. Results and Interpretation

## Continuous notebook saved results

From the notebook outputs:

| Model | One-step MSE | Forecast MSE |
|---|---:|---:|
| `RNN` | `0.079916` | `0.451069` |
| `LSTM` | `0.079553` | `0.321931` |
| `LSTMTransformer` | `0.006444` | `0.358934` |

Forecast horizon:

- `H_forecasting = 20`

Interpretation that is supported:

- the hybrid model is best on one-step MSE
- the plain `LSTM` is best on the saved 20-step forecast MSE
- one-step quality and long-horizon quality are not identical

## Continuous training loss logs

Saved logs:

- `RNN`: `1.411073 -> 0.131418`
- `LSTM`: `1.756535 -> 0.117281`
- hybrid Transformer: `1.881302 -> -0.468759`

Important oral point:

- Gaussian NLL can become negative
- that is not automatically a bug

## Discrete notebook saved results

From the notebook outputs:

| Model | One-step CE | One-step MSE | Autoregressive MSE |
|---|---:|---:|---:|
| `Discrete RNN` | `0.982942` | `0.040340` | `0.672988` |
| `Discrete LSTM` | `1.432115` | `0.062142` | `0.515349` |
| `Discrete LSTM-Transformer` | `0.060391` | `0.003626` | `0.228018` |

Autoregressive horizon:

- `horizon = T - T_train = 100`

Strict caveat:

- the saved discrete autoregressive MSE averages over `num_sequences=64`

Supported interpretation:

- the hybrid discrete model is clearly best in the saved discrete experiment
- the discrete `LSTM` underperforms the discrete `RNN` in the saved outputs

## What the metrics mean

### Cross-entropy

- lower means the model assigns higher probability to the correct next bin

### One-step MSE

- lower means better average squared error for one-step prediction in value space

### Autoregressive MSE

- lower means better multi-step rollout quality after feeding predictions back in

## What can and cannot be concluded

Reasonable conclusion:

- the tokenized formulation works and performs strongly in this saved experiment

Unreasonable conclusion:

- the discrete formulation is proven generally superior

Why that stronger claim fails:

- losses differ
- training setups differ
- forecast horizons differ
- discrete autoregressive evaluation uses 64 sequences, not the full test set

## Main limitations

- normalization leakage
- quantization-bound leakage in the discrete notebook
- no validation split
- no repeated-seed analysis
- continuous “Transformer” is actually `LSTMTransformer`
- not all final metrics are computed on identical horizons or identical evaluation coverage

# 8. Likely Professor Questions

These are intentionally harder than the first draft.

1. **What exact probabilistic object does the continuous model learn?**  
   A Gaussian conditional distribution at each time step, parameterized by a predicted `mean` and a positive `std` from `RNN.forward(...)`, `LSTM.forward(...)`, or `LSTMTransformer.forward(...)`.

2. **What exact probabilistic object does the discrete model learn?**  
   A categorical distribution over 128 bins at each time step, represented by logits of shape `(B, T, 128)` and trained with `token_cross_entropy_loss(...)`.

3. **Why is `Y` shifted by one step relative to `X`?**  
   Because the task is one-step-ahead forecasting. The code explicitly builds `X = seq[:-1]` and `Y = seq[1:]`, so every position learns next-step prediction.

4. **Where exactly is train/test leakage introduced?**  
   In both notebooks, the full series is standardized before splitting. In the discrete notebook, `train_min` and `train_max` are also computed from the full subsampled series.

5. **Why is the variable name `train_min` misleading?**  
   Because the code sets `train_min = series_sub.min()` before any train/test split, so it is not actually a training-only bound.

6. **How is the continuous training loop different from the discrete one?**  
   The continuous notebook uses full-batch optimization over all training sequences at once, while the discrete notebook uses minibatches from a `DataLoader`.

7. **Does the continuous training loss use sampled outputs or predicted means?**  
   For the final saved baseline runs, the loss uses `mean` and `std` through Gaussian NLL. The sampled trajectory is returned by `forward(...)` but not used in the final `RNN_extended` or `LSTM_extended` loss.

8. **What confusion can happen around `TR_extended`?**  
   `TR_extended` is defined twice in `AR_RNN-v8.ipynb`. The earlier version uses L2 loss on sampled outputs, but it is overwritten by a later Gaussian-NLL version, and the later one is the one actually instantiated.

9. **Is the reported baseline Transformer a pure Transformer?**  
   No. The class is `LSTMTransformer`, so it first uses an LSTM and then a Transformer encoder.

10. **Why use `softplus` on the standard deviation output?**  
   To force the predicted standard deviation to be positive in a smooth differentiable way.

11. **Why can Gaussian NLL become negative?**  
   Because continuous densities can exceed 1 for narrow distributions, which can make log-likelihood positive and negative log-likelihood negative.

12. **What exact tensor reshaping happens in `token_cross_entropy_loss(...)`?**  
   Logits `(B, T, C)` are reshaped to `(B*T, C)` and targets `(B, T)` are reshaped to `(B*T,)`, so cross-entropy is applied across all sequence positions jointly.

13. **How is one-step MSE computed in the discrete notebook?**  
   Not by argmax. The notebook computes the expected bin index under the softmax distribution, dequantizes that expectation, and compares it to dequantized targets.

14. **Why is that different from standard classification evaluation?**  
   Because the final one-step MSE is measured in reconstructed value space, not just in token accuracy or argmax-bin accuracy.

15. **What exactly happens inside `sample_tokens(...)`?**  
   The model runs on the full current token sequence, takes the logits at the last position, divides by temperature, applies softmax, samples a token with `torch.multinomial`, appends it, and repeats.

16. **Is the discrete autoregressive forecast deterministic?**  
   No. It is stochastic because it samples tokens with `torch.multinomial(...)`.

17. **What exactly is averaged in the discrete autoregressive MSE?**  
   The mean MSE across `num_sequences=64` sampled test windows, not across all 1000 test windows.

18. **What exact horizon is used in the continuous notebook?**  
   `H_forecasting = 20`.

19. **What exact horizon is used in the discrete notebook?**  
   `horizon = T - T_train = 100`, then clipped by `min(...)`, which stays 100 in the saved setup.

20. **Why are the continuous and discrete horizon metrics not directly comparable?**  
   Because they are measured over different forecast lengths, under different losses, and not even over the same number of evaluated sequences in the autoregressive case.

21. **How does the RNN/LSTM autoregressive rollout differ from the hybrid Transformer rollout?**  
   RNN/LSTM reuse recurrent hidden state and feed the last sampled value back in. The hybrid Transformer rebuilds a full sliding input window each step and takes the sampled last output as the new appended point.

22. **What shape does the continuous model receive during training?**  
   `(N_train, T_train, 1)`.

23. **What shape does the discrete model receive during training?**  
   `(batch_size, 200)` tokens per batch.

24. **What does `TokenSequenceDataset` contribute beyond a plain array?**  
   It converts `X` and `Y` to `torch.long` tensors and makes them compatible with `DataLoader`.

25. **Why is `T_train = 100` a potential oral-exam trap in the discrete notebook?**  
   Because it is presented as the context length, but training still uses full length-200 sequences from `train_loader`. It is not a strict training truncation parameter there.

26. **What is one concrete code field that is stored but not actually used in discrete training?**  
   `DiscreteRNNExtended.sequence_length` is stored in the object but not used inside `DiscreteRNNExtended.trainloop(...)`.

27. **Could the discrete model represent multimodal uncertainty better than a single Gaussian?**  
   In principle, yes, because a categorical distribution over bins can place mass on multiple separated bins. The repo motivates this idea, but it does not include a formal calibration study proving it.

28. **Why should we be careful saying the discrete method is better?**  
   Because the experimental setups are not strictly matched and the autoregressive metric coverage differs.

29. **How do you know the models are not overfitting?**  
   We do not know well from this repo alone, because there is no validation split or repeated-seed experiment. We only know the saved test metrics.

30. **If the professor asks for the weakest part of the experimental design, what should you say first?**  
   Train/test leakage in preprocessing and the lack of a validation split.

31. **If the professor asks for the weakest part of the result comparison, what should you say first?**  
   The continuous and discrete results are not matched on forecast horizon or full-test autoregressive coverage.

32. **Why might the hybrid baseline have the best one-step MSE but not the best horizon MSE?**  
   Because local next-step fit and long-rollout stability are different. Error compounds during autoregression.

33. **What exactly is the role of positional encoding in the hybrid discrete model?**  
   After the LSTM output is projected to `d_model`, positional encoding injects explicit time-step order before the Transformer encoder.

34. **Why is the causal mask necessary in the discrete Transformer models?**  
   Without it, a token at position `t` could attend to future positions and leak target information during training.

35. **What would be the cleanest follow-up experiment?**  
   Re-run continuous and discrete models with train-only preprocessing, a validation split, matched horizons, repeated seeds, and autoregressive evaluation over the full test set.

## Red-Flag Questions

These are the questions a professor may ask if they think you only memorized a summary.

1. **Show me exactly where the target shift is created in code.**
2. **Show me exactly where positivity of `std` is enforced.**
3. **Why is the discrete one-step MSE not just token accuracy?**
4. **Why is the discrete autoregressive metric not a full-test-set metric?**
5. **What is the difference between `mean`, `std`, `hidden`, and `sample` in the continuous models?**
6. **What is the exact input shape to `LSTMTransformer.forward(...)`?**
7. **What is the exact shape of the Transformer logits in the discrete notebook?**
8. **Why is calling the baseline model “Transformer” slightly misleading?**
9. **Which class is defined twice, and why does that matter?**
10. **Why is `train_min` not really a training-only statistic?**
11. **Why does `T_train` mean different things in the two notebooks?**
12. **What in the repo would you change first to make the experiment publishable rather than classroom-level?**

### Red-Flag Answers

1. **Show me exactly where the target shift is created in code.**  
   In `AR_RNN-v8.ipynb`, the shift is created when the window `seq` is split into `X_list.append(seq[:-1])` and `Y_list.append(seq[1:])`. In `STUDENT_discrete_time_series_llm_notebook.ipynb`, it is created by `X = np.stack([series_bins[s:s+T] ...])` and `Y = np.stack([series_bins[s+1:s+T+1] ...])`.

2. **Show me exactly where positivity of `std` is enforced.**  
   In the continuous notebook, `std` is made positive with `torch.nn.functional.softplus(...)`. That appears in `RNN.forward(...)`, `LSTM.forward(...)`, and `LSTMTransformer.forward(...)`.

3. **Why is the discrete one-step MSE not just token accuracy?**  
   Because `evaluate_one_step(...)` does not score only whether the predicted bin is correct. It computes a probability distribution over bins, takes the expected bin index under that distribution, dequantizes that expected value, and then computes MSE in continuous value space.

4. **Why is the discrete autoregressive metric not a full-test-set metric?**  
   Because `evaluate_autoregressive(...)` is called with `num_sequences=64`, and inside the function it uses `num_sequences = min(num_sequences, len(X_test))`. So the reported autoregressive MSE is averaged over 64 test windows, not all 1000.

5. **What is the difference between `mean`, `std`, `hidden`, and `sample` in the continuous models?**  
   `mean` is the predicted Gaussian center at each time step. `std` is the predicted Gaussian spread at each time step. `hidden` is the recurrent state summary returned by the RNN/LSTM backbone, or contextual representation in the hybrid case. `sample` is a stochastic draw from the predicted Gaussian, computed as `mean + std * noise`.

6. **What is the exact input shape to `LSTMTransformer.forward(...)`?**  
   `(batch_size, seq_len, 1)`. In training and evaluation, the notebook constructs tensors like `(N_train, T_train, 1)` or `(N_test, T_test, 1)` before calling the model.

7. **What is the exact shape of the Transformer logits in the discrete notebook?**  
   `(B, T, 128)`, because the output head maps each time step to `NUM_BINS = 128` classes.

8. **Why is calling the baseline model “Transformer” slightly misleading?**  
   Because the baseline class is `LSTMTransformer`, not a pure Transformer. It first runs an LSTM, then projects the LSTM outputs, adds positional encoding, and only then applies Transformer encoder layers.

9. **Which class is defined twice, and why does that matter?**  
   `TR_extended` is defined twice in `AR_RNN-v8.ipynb`. The first version trains with L2 loss on sampled outputs, but the second version overwrites it and trains with Gaussian NLL. This matters because the saved final Transformer-hybrid run uses the second definition, so that is the loss we should describe.

10. **Why is `train_min` not really a training-only statistic?**  
   Because the code sets `train_min = series_sub.min()` and `train_max = series_sub.max()` before any train/test split. So those bounds are computed from the entire subsampled series.

11. **Why does `T_train` mean different things in the two notebooks?**  
   In the continuous notebook, `T_train = 100` is directly used to slice training inputs and targets, so models are trained on the first 100 positions. In the discrete notebook, `T_train = 100` is mainly the context length used for forecasting and evaluation, while training still uses full 200-token sequences from `train_loader`.

12. **What in the repo would you change first to make the experiment publishable rather than classroom-level?**  
   First, fix preprocessing leakage by fitting normalization and quantization bounds on training data only. Second, add a validation split and repeated-seed runs. Third, make the comparisons fairer by matching forecast horizons and evaluating autoregressive metrics over the full test set for all models.

# 9. "If You Get Stuck" Answers

Use these only when needed, and keep them honest.

- "The code supports that interpretation, but not a stronger claim."
- "That part is implemented in the notebook, but the experiment is not fully controlled."
- "A key limitation is that preprocessing is not train-only."
- "The tradeoff is flexibility versus precision, especially in the quantized model."
- "The hybrid model appears stronger here, but the repo does not isolate exactly which component causes the gain."
- "That result is saved in the notebook output, but it is measured under a different horizon from the baseline."
- "The discrete model's long-horizon metric is averaged over 64 test sequences, not the full test set."
- "If we wanted a stronger answer, we would need a validation split and repeated runs."

# 10. 2-Minute Presentation Script

This repository compares two ways to do probabilistic forecasting on the `OT` signal from the ETTh1 dataset. In both notebooks, the `OT` series is standardized, subsampled by 10, and broken into many random shifted windows of length 200 so that each input position predicts the next position.

The first notebook, `AR_RNN-v8.ipynb`, uses continuous models: an `RNN`, an `LSTM`, and a hybrid `LSTMTransformer`. These models take input tensors shaped like `(N, T, 1)` and predict a Gaussian mean and standard deviation at each time step. They are trained mainly with Gaussian negative log-likelihood and evaluated with one-step MSE and autoregressive horizon MSE.

The second notebook, `STUDENT_discrete_time_series_llm_notebook.ipynb`, first quantizes the normalized signal into 128 bins and turns forecasting into next-token prediction. Models like `DiscreteRNN`, `DiscreteLSTM`, and `DiscreteLSTMTransformer` take integer token sequences and output logits over 128 bins. They are trained with cross-entropy, then evaluated both with token-level cross-entropy and with MSE after dequantizing predictions back to value space.

The main conclusion supported by the repo is that the tokenized formulation works and can be very strong in this experiment, especially with the hybrid `DiscreteLSTMTransformer`. But the repo also has important limitations: preprocessing leakage, no validation split, a hybrid baseline that is called “Transformer,” and non-matched autoregressive evaluations across the two notebooks.

# 11. Individual Teammate Study Guide

## A. Data / preprocessing owner

Must understand:

- `df["OT"]` extraction
- full-series standardization
- why that causes leakage
- subsampling by 10
- how shifted windows are built
- `T = 200`, `N = 5000`, `T_train = 100`
- discrete quantization and dequantization
- why `train_min` / `train_max` are not really training-only statistics

## B. Model architecture owner

Must understand:

- `RNN`, `LSTM`, `LSTMTransformer`
- `DiscreteRNN`, `DiscreteLSTM`, `DiscreteTransformer`, `DiscreteLSTMTransformer`
- exact tensor shapes through each forward pass
- why `softplus` is used
- why positional encoding is needed
- why the causal mask is needed
- why the hybrid models are not pure Transformers

## C. Training / evaluation owner

Must understand:

- `RNN_extended.trainloop(...)`
- `LSTM_extended.trainloop(...)`
- both versions of `TR_extended`
- `train_discrete_model(...)`
- `token_cross_entropy_loss(...)`
- `gaussian_nll(...)`
- full-batch vs minibatch difference
- one-step evaluation vs autoregressive rollout
- the 64-sequence caveat in discrete autoregressive evaluation

## D. Results / limitations owner

Must understand:

- exact saved metrics from both notebooks
- why one-step and long-horizon results can differ
- why the hybrid discrete model is notable
- why the comparisons are not fully controlled
- leakage, no validation split, no repeated seeds

## E. Code walkthrough owner

Must understand:

- where each key class/function lives
- which notebook cells define data, models, training, and evaluation
- that `TR_extended` is defined twice
- that `DiscreteTransformer` is defined but not used in final comparison
- that saved notebook outputs are part of the evidence base

# Final Oral Advice

If you only remember five things, remember these:

1. The repo compares continuous Gaussian forecasting against discrete token forecasting on the same `OT` signal.
2. Both notebooks use shifted windows, so the task is one-step-ahead prediction at every sequence position.
3. The baseline “Transformer” is actually a hybrid `LSTMTransformer`.
4. The discrete autoregressive metric is averaged over 64 test sequences, not the full test set.
5. The biggest weaknesses are preprocessing leakage, no validation split, and not-fully-matched evaluations.
