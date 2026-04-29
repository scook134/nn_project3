# Technical Report: Discrete Probabilistic Time-Series Forecasting on ETTh1

## 1. Introduction

Classical probabilistic forecasting methods often assume that each future value is generated from a Gaussian distribution parameterized by a predicted mean and variance. This is a practical choice, but it imposes a strong structural assumption: the predictive distribution is unimodal and symmetric. For time-series problems with ambiguous or multi-trajectory futures, that assumption can be restrictive. A Gaussian predictor can widen its variance to express uncertainty, but it still cannot represent distinctly different plausible outcomes as separate modes.

The discrete forecasting notebook reframes the problem as categorical sequence modeling. Instead of directly predicting a continuous value, the normalized signal is quantized into 128 bins, and the model predicts the next bin index. In this formulation, each time step becomes a token, and forecasting becomes next-token prediction over a finite vocabulary.

This creates a direct connection to language modeling. In the same way that a language model predicts the next word or subword token conditioned on previous tokens, the discrete time-series model predicts the next quantized state conditioned on previous quantized states. The project therefore connects DeepAR-style autoregressive forecasting with modern sequence-modeling ideas drawn from LLMs.

## 2. Dataset and Preprocessing

Both notebooks use the `ETTh1.csv` dataset and select the univariate `OT` column as the forecasting target. In the original notebook this is explicitly described as the oil temperature signal.

The preprocessing pipeline implemented in both notebooks is:

- Load `ETTh1.csv`.
- Extract `df["OT"]` as a `float32` NumPy array.
- Standardize the full series using its global mean and standard deviation.
- Subsample the standardized signal with `subsample_factor = 10`.

The saved notebook output reports:

- Total length after subsampling: `1742`

After subsampling, both notebooks construct random overlapping windows:

- Sequence length: `T = 200`
- Number of sampled windows: `N = 5000`
- Train/test split ratio: `0.8 / 0.2`
- Training windows: `4000`
- Test windows: `1000`
- Conditioning context length: `T_train = 100`

The Gaussian notebook samples random start indices using `np.random.randint`, while the discrete notebook uses `rng = np.random.default_rng(0)` and `rng.integers(...)`. In both cases, the input and target sequences are aligned one step apart:

- `X`: the first `T` points of each window
- `Y`: the next `T` points of each window

This creates the standard autoregressive supervision pattern:

- `X[:, t]` predicts `Y[:, t] = X[:, t+1]`

## 3. Quantization

The discrete notebook uses uniform quantization with:

- `NUM_BINS = 128`

The implemented quantization function is:

- Scale the normalized value into `[0, 1]`
- Multiply by `num_bins`
- Take `floor(...)`
- Clip indices into `[0, 127]`

Formally, the code computes:

`bins = floor(num_bins * (x - x_min) / (x_max - x_min + 1e-8))`

then clips to the valid token range.

The implemented dequantization function maps each token back to its bin center:

- `centers = (bins + 0.5) / num_bins`
- `x = x_min + centers * (x_max - x_min)`

In the saved notebook, the quantization bounds are set as:

- `train_min = series_sub.min()`
- `train_max = series_sub.max()`

Although the notebook comments say that a good practice would be to compute boundaries from the training split only, the executed notebook uses the full subsampled series range.

The saved notebook output reports:

- Unique bins used: `121`
- First 20 bin indices: `[89, 62, 56, 63, 64, 73, 80, 60, 66, 85, 89, 85, 85, 89, 89, 79, 106, 86, 91, 91]`

## 4. Model Design

The discrete notebook implements four token-based architectures:

- `DiscreteRNN`
- `DiscreteLSTM`
- `DiscreteTransformer`
- `DiscreteLSTMTransformer`

The saved training and evaluation cells actually run:

- `DiscreteRNN`
- `DiscreteLSTM`
- `DiscreteLSTMTransformer`

The pure `DiscreteTransformer` is implemented but not included in the final training/evaluation block.

All discrete models share the same basic design pattern:

- Integer token inputs
- Learned embedding layer `nn.Embedding(num_bins, d_model)`
- Sequence model over embeddings
- Output projection to `128` logits per time step

The architecture-specific adaptations are:

- `DiscreteRNN`: embedding -> `nn.RNN(..., nonlinearity="relu")` -> linear head
- `DiscreteLSTM`: embedding -> `nn.LSTM(...)` -> linear head
- `DiscreteTransformer`: embedding -> sinusoidal positional encoding -> causal Transformer encoder -> linear head
- `DiscreteLSTMTransformer`: embedding -> LSTM -> linear projection -> sinusoidal positional encoding -> causal Transformer encoder -> linear head

The causal Transformer masking function is implemented as:

- upper-triangular boolean mask with `torch.triu(..., diagonal=1).bool()`

The output of every discrete model is a tensor of logits with shape `(batch, seq_len, 128)`. These logits define a categorical distribution over the 128 bins at each time step.

The original Gaussian notebook implements three continuous-output probabilistic models:

- `RNN`
- `LSTM`
- `LSTMTransformer`

These Gaussian models predict:

- a mean
- a positive standard deviation via `softplus`

and sample continuous values from the corresponding Gaussian at each step.

## 5. Training Procedure

The discrete notebook trains next-token predictors with cross-entropy loss. The token loss function reshapes logits from `(B, T, C)` to `(B*T, C)` and targets from `(B, T)` to `(B*T)`, then applies `F.cross_entropy(...)`.

The training setup actually used in the saved notebook is:

- Batch size: `64`
- Train batches: `63`
- Test batches: `16`
- Optimizer: Adam
- Learning rate: `1e-3`
- Epochs: `12`
- Gradient clipping: `max_norm = 2.0`

The discrete training inputs and targets are aligned one step apart across the full token sequence:

- input sequence: `X`
- target sequence: `Y`

This means the model is trained autoregressively at every position in the window, not only at the final position.

Two training wrappers are used:

- `DiscreteRNNExtended.trainloop(...)`
- `train_discrete_model(...)` for the LSTM and LSTM-Transformer

The saved training losses printed by the notebook are:

- Discrete RNN: epoch 1 `4.380498`, epoch 12 `0.993823`
- Discrete LSTM: epoch 1 `4.473671`, epoch 12 `1.472224`
- Discrete LSTM-Transformer: epoch 1 `4.232699`, epoch 12 `0.085779`

In the original Gaussian notebook, the training procedures differ by model family:

- Gaussian RNN: full-batch training, Gaussian negative log-likelihood, `250` iterations, Adam, learning rate `0.005`, gradient clipping `2.0`
- Gaussian LSTM: full-batch training, Gaussian negative log-likelihood, `250` iterations, Adam, learning rate `0.005`, gradient clipping `2.0`
- Gaussian LSTM-Transformer: the final executed `TR_extended` definition uses Gaussian negative log-likelihood, `250` iterations, Adam, learning rate `0.005`, gradient clipping `2.0`

The saved Gaussian training logs report:

- RNN: iteration 0 `1.411073`, iteration 200 `0.131418`
- LSTM: iteration 0 `1.756535`, iteration 200 `0.117281`
- Transformer: iteration 0 `1.881302`, iteration 200 `-0.468759`

## 6. Forecasting Method

The discrete notebook performs forecasting by sequential token sampling. The implemented `sample_tokens(...)` function:

- starts from a seed token sequence
- runs the model on the current sequence
- extracts the logits at the final position
- divides by temperature
- converts logits to probabilities with `softmax`
- samples the next token with `torch.multinomial`
- appends the sampled token to the sequence

This loop is repeated for the desired number of future steps.

For long-horizon evaluation, the notebook:

- uses the first `T_train = 100` tokens of a test window as context
- samples `horizon = T - T_train = 100` additional tokens
- converts sampled future tokens back to continuous values via `tokens_to_values(...)`

The forecast is generative because future predictions are not fixed deterministic regressions from the original context alone. Each sampled token is fed back into the model and influences subsequent predictions. As a result, the model produces full sampled trajectories rather than only point estimates.

The Gaussian notebook also uses autoregressive rollout, but it rolls forward continuous sampled values from Gaussian outputs instead of categorical token samples. For the saved evaluation cells:

- one-step evaluation uses `T_test = 100`
- autoregressive forecasting horizon uses `H_forecasting = 20`
- effective horizon is `H_eff = 20`

## 7. Evaluation and Results

### Discrete notebook results

The discrete notebook evaluates:

- one-step cross-entropy
- one-step MSE after mapping predictions back to continuous values
- autoregressive MSE after sampling future tokens and decoding them

The saved final comparison table reports:

| Model | One-step Cross-Entropy | One-step MSE | Autoregressive MSE |
|---|---:|---:|---:|
| Discrete LSTM-Transformer | 0.060391 | 0.003626 | 0.228018 |
| Discrete LSTM | 1.432115 | 0.062142 | 0.515349 |
| Discrete RNN | 0.982942 | 0.040340 | 0.672988 |

### Gaussian notebook results

The original notebook reports the following final metrics:

| Model | One-step MSE | Horizon MSE (H=20) |
|---|---:|---:|
| RNN | 0.079916 | 0.451069 |
| LSTM | 0.079553 | 0.321931 |
| Transformer | 0.006444 | 0.358934 |

The Gaussian notebook’s “Transformer” result comes from the implemented `LSTMTransformer` / `TR_extended` hybrid model rather than a pure Transformer-only architecture.

### Comparison

From the saved notebook outputs:

- Among the discrete models actually trained, the `Discrete LSTM-Transformer` is the strongest on both one-step and autoregressive metrics.
- Among the Gaussian models, the hybrid Transformer has the best one-step MSE, while the Gaussian LSTM has the best horizon MSE at `H=20`.

Direct numeric comparison should be interpreted carefully because the long-horizon setups are not identical:

- discrete notebook autoregressive horizon: `100` steps
- Gaussian notebook autoregressive horizon: `20` steps

That means the autoregressive MSE values are not on exactly the same evaluation horizon.

### Prediction quality and uncertainty behavior

The notebooks provide the following qualitative artifacts:

- Gaussian notebook:
  - plots of predictive mean with `± 2σ` error bars for one-step prediction
  - plots of autoregressive reconstructions beyond the conditioning window
- Discrete notebook:
  - training-loss curves
  - one-step plots comparing true values against decoded expected predictions
  - autoregressive plots comparing the true trajectory against sampled forecasts

The notebooks do not provide a dedicated calibration metric, likelihood comparison across both paradigms, or a quantitative measure of multimodality. For those quantities, the appropriate entries are:

- Quantitative uncertainty calibration comparison: `[INSERT RESULT]`
- Explicit multimodality metric: `[INSERT RESULT]`
- Asymmetry/non-Gaussianity diagnostic: `[INSERT RESULT]`

The saved outputs do support the following evidence-based summary:

- the discrete formulation is fully operational
- token sampling produces generative trajectories
- the hybrid discrete LSTM-Transformer performs best among the discrete models that were actually trained

### Strengths and weaknesses of the discrete approach

Strengths supported by the implementation:

- categorical outputs can represent flexible non-Gaussian predictive structure
- autoregressive sampling yields genuinely generative future trajectories
- the hybrid sequence model performs strongly in the saved experiments

Weaknesses visible from the notebook design:

- quantization introduces approximation error
- performance depends on the chosen bin resolution
- sampled forecasts introduce stochastic variance
- the saved notebook does not report calibration-specific uncertainty metrics

## 8. Discussion

The main conceptual advantage of discrete forecasting is representational flexibility. A categorical distribution over bins can approximate:

- multimodal uncertainty
- asymmetric uncertainty
- non-Gaussian predictive shapes

This is difficult for a single Gaussian output head, which is constrained to one symmetric bell-shaped mode at each step.

The tradeoffs are equally important:

- Quantization error: mapping a continuous signal into 128 bins loses precision.
- Bin resolution: too few bins cause coarse reconstructions; too many bins increase vocabulary size and learning difficulty.
- Sampling variance: autoregressive token sampling introduces run-to-run variability in long-horizon forecasts.
- Computational cost: categorical output over 128 classes can be more expensive than predicting only mean and variance, especially as the vocabulary grows.

The notebooks also show an architectural tradeoff. Simple recurrent models work in the discrete setting, but the best saved results come from the hybrid LSTM-Transformer, suggesting that local sequential structure plus richer context modeling is beneficial after tokenization.

## 9. Conclusion

This project implements a full discrete probabilistic forecasting pipeline on the ETTh1 `OT` signal:

- preprocessing and normalization
- uniform quantization into 128 bins
- autoregressive token-sequence training
- discrete RNN, LSTM, Transformer, and LSTM-Transformer model definitions
- sequential token sampling for forecasting
- decoding token forecasts back into continuous values for evaluation

It also preserves the original Gaussian forecasting baseline notebook with autoregressive RNN, LSTM, and hybrid LSTM-Transformer models.

Taken together, the two notebooks connect two important views of probabilistic forecasting. The original notebook follows the DeepAR-style idea of forecasting by autoregressively predicting Gaussian parameters. The discrete notebook replaces that continuous output assumption with token prediction, bringing the problem closer to modern LLM-style sequence modeling. The result is a concrete demonstration that time-series forecasting can be reformulated as next-token generation while retaining probabilistic, autoregressive behavior.
