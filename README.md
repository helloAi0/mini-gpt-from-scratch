# 🧠 Mini GPT From Scratch

A minimal GPT-style language model implemented from scratch using PyTorch.

This project recreates the core components of a Transformer-based language model, including:

- Token embeddings  
- Positional embeddings  
- Multi-head self-attention  
- Transformer blocks  
- Layer normalization  
- Training loop with gradient clipping  
- Text generation  

---

## 📂 Project Structure

mini-gpt-from-scratch/
│
├── src/
│   ├── model.py        # GPT model definition
│   ├── attention.py    # Transformer blocks & attention
│   ├── dataset.py      # Data loading & batching
│   ├── train.py        # Training loop
│   ├── generate.py     # Text generation
│   └── config.py       # Hyperparameters
│
├── requirements.txt
└── README.md

---

## ⚙️ Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt