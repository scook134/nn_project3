📊 Discrete Probabilistic Time Series Forecasting
🎯 Overview

This project explores an alternative approach to time series forecasting by replacing traditional Gaussian output distributions with a discrete probabilistic formulation.

Instead of modeling predictions as continuous values:

xₜ ~ N(μₜ, σₜ²)

we transform the problem into a categorical prediction task:

xₜ ~ Categorical(p₁, …, p₁₂₈)

By discretizing the signal into bins, the time series becomes a sequence of tokens, similar to natural language.

👉 This turns forecasting into a next-token prediction problem, analogous to modern language models.

🧠 Key Idea

After quantization:

Each time step → a token
The model learns:
P(next token | previous tokens)

This mirrors how language models like GPT operate:

Time Series	Language Model
signal value	word/token
sequence	sentence
forecasting	next-token prediction
🚀 Objectives
Understand limitations of Gaussian forecasting
Implement discrete probabilistic models
Use embeddings for time series
Train autoregressive sequence models
Generate forecasts via sampling
Compare continuous vs discrete uncertainty
🧩 Project Structure
.
├── notebook.ipynb        # Main implementation
├── ETTh1.csv             # Dataset
├── README.md             # Project documentation
⚙️ Methodology
1. Preprocessing
Load ETTh1 dataset (OT signal)
Normalize values using StandardScaler
Create sliding windows for training
2. Quantization
Discretize signal into 128 uniform bins
Convert values → token indices
3. Modeling
Replace scalar inputs with embedding vectors
Use LSTM-based autoregressive model
Output: logits over 128 categories
4. Training
Loss: Cross-Entropy
Objective: predict next token in sequence
5. Forecasting
Generate predictions via autoregressive sampling
Convert tokens back to continuous values
6. Evaluation
Visual comparison of:
True future
Sampled forecasts
Analyze uncertainty via multiple sampled trajectories
📈 Results & Insights
✅ Advantages of Discrete Modeling

Unlike Gaussian models, this approach can represent:

Multi-modal distributions (multiple possible futures)
Asymmetric uncertainty
Complex, non-linear behaviors
⚠️ Limitations
Quantization introduces information loss
Requires careful choice of:
number of bins
scaling strategy
🔍 Example Output
Single forecast vs ground truth
Multiple sampled trajectories showing uncertainty spread
🧪 Future Improvements
🔁 Replace LSTM with Transformer (GPT-style)
🌡️ Add temperature / top-k sampling
🔢 Use non-uniform quantization
📊 Increase vocabulary size (e.g., 256 bins)
📡 Add exogenous variables
📉 Compare quantitatively with Gaussian baselines
📦 Requirements
pip install numpy pandas matplotlib torch scikit-learn
▶️ How to Run
Place ETTh1.csv in the project directory
Open the notebook:
jupyter notebook notebook.ipynb
Run cells sequentially
🔥 Why This Matters

Traditional forecasting assumes:

single peak distributions
symmetric uncertainty

This project demonstrates a shift toward:

👉 language-model-style forecasting

where uncertainty is:

flexible
expressive
data-driven
📚 References
Sequence modeling & language models
Probabilistic forecasting literature
Transformer architectures
