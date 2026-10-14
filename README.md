# Automated Source Code Summarizer and Comment Generator 🤖⚡

An end-to-end Deep Learning application that generates natural language summaries and docstrings from source code snippets using Pre-trained Transformer models (**CodeT5 / T5**).

---

## 📌 Project Overview
- **Domain:** Natural Language Processing (NLP) / Code Intelligence
- **Task:** Code-to-Text Generation (Source Code Summarization)
- **Model Architecture:** Sequence-to-Sequence (Seq2Seq) Transformer with Encoder-Decoder Multi-Head Attention
- **Frontend & Backend:** Streamlit (Single-file lightweight Python web app)

---

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Deep Learning Framework:** PyTorch (`torch`)
- **Model Hub & Tokenizers:** Hugging Face `transformers`
- **Pre-trained Model Checkpoint:** `Salesforce/codet5-base-multi-sum` (trained on CodeSearchNet across Python, JavaScript, Java, Go, PHP, Ruby)
- **Web Interface:** Streamlit

---

## 🚀 How to Run the Project

### 1. Open Terminal / PowerShell
Navigate to the project folder:
```powershell
cd "c:\Users\abhij\OneDrive\Desktop\Deep Learning Project"
```

### 2. Install Required Dependencies
*(If not already installed)*
```powershell
pip install -r requirements.txt
```

### 3. Launch the Application
Run the Streamlit application:
```powershell
streamlit run app.py
```

Your browser will automatically open:
```
Local URL: http://localhost:8501
```

---

## 💡 How It Works (Step-by-Step)

1. **Input Code:** You can paste any function (Python, JS, Java, etc.) or choose from the built-in dropdown presets.
2. **Subword Tokenization:** The code is converted into subword tokens using Byte-Pair Encoding (BPE) / SentencePiece.
3. **Transformer Encoder:** Processes code tokens and builds dense contextual vector representations with multi-head self-attention.
4. **Transformer Decoder:** Autoregressively generates the English summary token by token via cross-attention with beam search decoding.
5. **Comment & Docstring Injection:** The generated explanation is automatically inserted as a formatted docstring (`"""..."""` or JSDoc `/** ... */`) inside the source code.
6. **Quantitative Evaluation Tab:** You can compare the model's generated output against a ground-truth reference summary to calculate live **BLEU** and **ROUGE** scores.

---

## 🎓 Viva & Presentation Q&A Guide

| Question | Answer to Give Your Examiner |
|---|---|
| **What DL architecture are you using?** | We use an **Encoder-Decoder Transformer** (specifically CodeT5). The encoder processes bidirectional code context, and the autoregressive decoder generates natural language word-by-word. |
| **Why not just standard LSTM or RNN?** | LSTMs suffer from vanishing gradients and struggle with long-range dependencies across distant code lines. Transformers use **self-attention** to directly relate any two tokens regardless of distance, and allow parallel training. |
| **What dataset was this model trained on?** | It was pre-trained and fine-tuned on **CodeSearchNet**, a benchmark dataset comprising over 2 million pairs of code functions and their corresponding human docstrings. |
| **How do you evaluate code summarization models?** | Using standard NLP metrics: **BLEU** (n-gram precision) and **ROUGE** (recall/F1 overlap), which compare generated summaries against ground-truth human descriptions. |
| **Why use Beam Search instead of Greedy Search?** | Greedy search picks only the highest probability token at each step, which often leads to sub-optimal sentences. Beam search tracks the top-$k$ most promising sequences in parallel, producing more coherent and fluent summaries. |
