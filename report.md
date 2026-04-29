# Discrete Probabilistic Forecasting for Time Series

This report summarizes the implementation and recorded outputs in [project3.ipynb](/C:/Users/LeoSp/OneDrive/Desktop/Neural Networks/nn_project3/project3.ipynb) for discrete probabilistic forecasting on the ETTh1 `OT` signal. Where the repository README states broader goals, this report distinguishes those goals from what is actually implemented in the notebook.

## 1. Introduction

Classical probabilistic forecasting often models the next value with a Gaussian distribution, for example by predicting a mean and variance for each step. That formulation is convenient, but it imposes a specific shape on predictive uncertainty: unimodal and symmetric around the mean. A discrete categorical formulation relaxes that assumption by converting a continuous signal into a sequence of tokens and predicting the next token with a categorical distribution over bins.

In this repository, the normalized `OT` signal is quantized into `128` bins, and forecasting is cast as next-token prediction. This is analogous to sequence modeling in language tasks in one narrow sense: the model learns a distribution for the next discrete symbol conditioned on previous symbols. The notebook therefore reframes time-series forecasting as autoregressive generation over quantized signal tokens rather than direct Gaussian parameter prediction.

## 2. Dataset and Preprocessing

The notebook loads `ETTh1.csv` and uses only the `OT` column as a univariate series ([project3.ipynb](/C:/Users/LeoSp/OneDrive/Desktop/Neural Networks/nn_project3/project3.ipynb)). The CSV in the repository contains `17420` rows. A plot titled `OT Signal (First 1000 points)` is generated from the first `1000` raw observations.

Preprocessing consists of standardization with `StandardScaler`, applied to the full `OT` series:

- Raw values are reshaped to `(-1, 1)`.
- `StandardScaler` is fit and applied.
- The standardized series is flattened back to one dimension.

The notebook then plots the first `1000` normalized points under the title `Normalized Signal`.

No explicit train/validation/test split is implemented in the notebook. The code creates training windows from the full tokenized sequence and feeds all windows into a single shuffled `DataLoader`. Validation set: `not available`. Test set: `not available`.

Windowing is defined as:

- Context length `SEQ_LEN = 96`
- Forecast generation length `PRED_LEN = 24`

The sequence construction function produces:

- `X[i] = tokens[i : i + 96]`
- `Y[i] = tokens[i + 1 : i + 97]`

The printed dataset shapes are:

| Item | Value |
|---|---:|
| `X.shape` | `(17300, 96)` |
| `Y.shape` | `(17300, 96)` |

This means the supervision target is a one-step-ahead shift over the full `96`-token window, not a separate `24`-step target tensor.

## 3. Discrete Forecasting Formulation

Quantization is implemented directly on the standardized signal:

- Number of bins: `128`
- Bin construction: uniform bins with `np.linspace(data_scaled.min(), data_scaled.max(), NUM_BINS + 1)`
- Tokenization: `np.digitize(...)-1`, then clipping to `[0, 127]`

The notebook also plots a histogram titled `Token Distribution`.

Under this formulation, each time step is represented by a token index in `{0, ..., 127}`. The model predicts logits over the `128` bins at each position. Training uses categorical cross-entropy over the predicted logits and shifted token targets.

To decode generated tokens back to continuous values, the notebook maps each token to the center of its corresponding bin and then applies `scaler.inverse_transform`. This yields forecasts in the original `OT` scale.

## 4. Models

Despite the broader modeling description in [README.md](/C:/Users/LeoSp/OneDrive/Desktop/Neural Networks/nn_project3/README.md), the notebook implements one model only: `TokenLSTM`.

The architecture is:

- Token embedding: `nn.Embedding(128, 64)`
- Recurrent backbone: `nn.LSTM(64, 128, batch_first=True)`
- Output projection: `nn.Linear(128, 128)`

The printed model summary is:

| Component | Configuration |
|---|---|
| Embedding | `Embedding(128, 64)` |
| LSTM | `LSTM(64, 128, batch_first=True)` |
| Output layer | `Linear(128, 128)` |

The scalar-valued time series is therefore replaced by discrete token indices, which are embedded before entering the LSTM. The final linear layer produces `128` logits per time step, corresponding to the categorical distribution over quantization bins.

Additional models:

| Model type | Status in notebook |
|---|---|
| Vanilla RNN | not available |
| GRU | not available |
| Transformer | not available |
| Gaussian baseline | not available |
| Multiple discrete architectures | not available |

## 5. Training Procedure

Training uses:

- Loss: `nn.CrossEntropyLoss()`
- Optimizer: `torch.optim.Adam`
- Learning rate: `1e-3`
- Batch size: `64`
- Epochs: `5`
- Device printed by the notebook: `cpu`

The training loop feeds each full input window into the LSTM, obtains logits for all `96` positions, reshapes logits to `(-1, 128)`, reshapes targets to `(-1)`, and applies cross-entropy. This is standard teacher-forced next-token training, since the model is trained against the shifted ground-truth sequence rather than its own sampled outputs.

The notebook prints the following average loss per epoch:

| Epoch | Printed loss |
|---|---:|
| 1 | 2.5773 |
| 2 | 1.9367 |
| 3 | 1.6499 |
| 4 | 1.3737 |
| 5 | 1.1553 |

Hardware runtime details beyond `cpu`: `not available`.

## 6. Forecast Generation

Forecast generation is implemented autoregressively. Given a starting token sequence:

1. The current context is passed through the model.
2. The logits at the last position are converted to probabilities with `softmax`.
3. The next token is sampled with `torch.multinomial`.
4. The sampled token is appended to the context after dropping the oldest token.
5. Steps 1 to 4 are repeated `24` times.

The resulting token sequence is then decoded to continuous values by mapping each token to its bin center and inverting the standardization.

Temperature sampling: `not available`.

Top-k or nucleus sampling: `not available`.

## 7. Evaluation and Results

The notebook provides qualitative plots and training loss, but no held-out quantitative forecasting metrics.

### Recorded numerical results

| Result | Value |
|---|---:|
| Number of rows in `ETTh1.csv` | 17420 |
| Number of bins | 128 |
| Context length | 96 |
| Forecast generation length | 24 |
| Window tensor shape | `(17300, 96)` |
| Target tensor shape | `(17300, 96)` |
| Batch size | 64 |
| Learning rate | 1e-3 |
| Epochs | 5 |
| Device | cpu |
| Final printed training loss | 1.1553 |

### Metrics and comparisons

| Item | Availability |
|---|---|
| Validation loss | not available |
| Test loss | not available |
| MAE | not available |
| MSE | not available |
| RMSE | not available |
| NLL on held-out data | not available |
| Calibration metric | not available |
| Coverage metric | not available |
| CRPS | not available |
| Comparison with Gaussian baseline | not available |

### Qualitative outputs present in the notebook

The notebook renders the following figures:

- `OT Signal (First 1000 points)`
- `Normalized Signal`
- `Token Distribution`
- `Discrete Probabilistic Forecast`
- `Multiple Sampled Futures (Uncertainty)`

The final two figures show:

- One sampled forecast plotted against the historical context and a decoded target sequence.
- Ten sampled futures plotted with transparency to visualize spread.

However, no numeric summary of forecast accuracy or uncertainty quality is printed alongside these figures.

One implementation detail is important for interpretation: in the evaluation cell, `true_future` is set to `tokens_to_values(Y[idx])`, and `Y[idx]` has length `96`, while generated forecasts have length `24`. The notebook therefore plots a `24`-step sampled forecast against a decoded shifted target window of length `96`. The notebook does not provide a separate numeric evaluation for the `24`-step horizon.

## 8. Discussion

The notebook demonstrates the main advantage of the discrete formulation clearly at the modeling level. Because the output is a categorical distribution over bins rather than a Gaussian parameter pair, the predictive distribution is not constrained to be symmetric or effectively unimodal. The multiple-sample plot is intended to illustrate this generative uncertainty view, although no formal calibration analysis is included.

At the same time, the implementation also reflects the standard limitations of this approach. Quantization necessarily discards some continuous precision, since all values inside a bin map to the same token and are decoded by the bin center. Because the bins are uniform, resolution is allocated evenly across the standardized range rather than according to where the data are most concentrated. The notebook also leaves open several factors that would likely matter in practice, including the number of bins, the sequence length, the model architecture, and the sampling strategy.

Another practical limitation is scope: the notebook implements only a single LSTM-based discrete model and reports only training loss plus qualitative plots. As a result, the repository demonstrates the formulation and a working generation pipeline, but it does not yet establish comparative forecasting performance against Gaussian baselines or alternative discrete architectures.

## 9. Conclusion

Within the scope of [project3.ipynb](/C:/Users/LeoSp/OneDrive/Desktop/Neural Networks/nn_project3/project3.ipynb), the project shows that univariate time-series forecasting on the ETTh1 `OT` signal can be reframed as autoregressive sequence modeling over quantized tokens. The implemented pipeline standardizes the signal, discretizes it into `128` uniform bins, trains an embedded LSTM with cross-entropy, and generates future trajectories by sequential sampling.

The notebook therefore provides a concrete proof of concept for discrete probabilistic forecasting as next-token prediction over a quantized time series. Quantitative forecasting performance on held-out data, Gaussian baseline comparisons, and calibration analysis are not available in the current repository state.
