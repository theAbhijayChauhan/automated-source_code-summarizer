import streamlit as st
import time
import ast
import re
from typing import Tuple, Dict, Any

# Page Configuration
st.set_page_config(
    page_title="Neural Code Summarizer & Comment Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        background-color: #E0E7FF;
        color: #3730A3;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Preset Code Examples for Quick Testing
# ---------------------------------------------------------
PRESETS = {
    "Select an example...": "",
    "Python: Binary Search": """def binary_search(arr, target):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1""",

    "Python: Merge Sort": """def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged""",

    "Python: Deep Learning Batch Normalizer": """def normalize_batch(tensor, epsilon=1e-5):
    mean = tensor.mean(dim=0, keepdim=True)
    var = tensor.var(dim=0, unbiased=False, keepdim=True)
    normalized = (tensor - mean) / (var + epsilon).sqrt()
    return normalized, mean, var""",

    "JavaScript: Fetch User Profile": """async function fetchUserProfile(userId) {
    try {
        const response = await fetch(`/api/users/${userId}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        return { success: true, user: data };
    } catch (error) {
        console.error("Failed to load user:", error);
        return { success: false, error: error.message };
    }
}""",

    "Java: Palindrome Checker": """public static boolean isPalindrome(String s) {
    String clean = s.replaceAll("[^a-zA-Z0-9]", "").toLowerCase();
    int left = 0;
    int right = clean.length() - 1;
    while (left < right) {
        if (clean.charAt(left) != clean.charAt(right)) {
            return false;
        }
        left++;
        right--;
    }
    return true;
}"""
}

# ---------------------------------------------------------
# Model Loading with Streamlit Cache
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(model_name: str):
    """Loads and caches Transformer tokenizer and model."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def generate_summary(
    code_snippet: str,
    tokenizer,
    model,
    num_beams: int = 4,
    max_length: int = 64,
    no_repeat_ngram_size: int = 2
) -> Tuple[str, float, list]:
    """Generates natural language summary for source code using Seq2Seq model."""
    import torch
    start_time = time.time()
    
    # Tokenize
    inputs = tokenizer(code_snippet, return_tensors="pt", max_length=512, truncation=True)
    input_ids = inputs["input_ids"]
    
    # Generate summary tokens using beam search
    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=no_repeat_ngram_size
        )
    
    latency = time.time() - start_time
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
    
    # Decode token breakdown for DL explanation
    token_tokens = [tokenizer.decode([t_id]) for t_id in input_ids[0][:40]]
    
    return summary_text, latency, token_tokens

def inject_docstring(code: str, summary: str, language: str) -> str:
    """Embeds generated summary as a docstring or header comment into the code."""
    summary_clean = summary.capitalize()
    if not summary_clean.endswith("."):
        summary_clean += "."

    if language.lower() == "python":
        lines = code.split("\n")
        # Find the function def line
        for idx, line in enumerate(lines):
            if re.match(r"^\s*def\s+[a-zA-Z_]\w*\s*\(", line):
                # Determine indentation of function body
                base_indent = len(line) - len(line.lstrip())
                inner_indent = " " * (base_indent + 4)
                docstring = f'{inner_indent}"""\n{inner_indent}{summary_clean}\n{inner_indent}"""'
                lines.insert(idx + 1, docstring)
                return "\n".join(lines)
        # If no def found, add docstring at the top
        return f'"""\n{summary_clean}\n"""\n' + code

    elif language.lower() in ["javascript", "typescript", "java", "c++", "c"]:
        jsdoc = f"/**\n * {summary_clean}\n */\n"
        return jsdoc + code
    else:
        return f"# {summary_clean}\n" + code

def compute_simple_bleu(reference: str, candidate: str) -> Dict[str, float]:
    """Computes basic unigram and bigram BLEU-style overlap for viva demo."""
    ref_tokens = re.findall(r"\w+", reference.lower())
    cand_tokens = re.findall(r"\w+", candidate.lower())
    
    if not ref_tokens or not cand_tokens:
        return {"BLEU-1": 0.0, "BLEU-2": 0.0, "ROUGE-1": 0.0}
    
    # Precision unigram
    overlap_unigram = sum(1 for w in cand_tokens if w in ref_tokens)
    p1 = overlap_unigram / len(cand_tokens) if cand_tokens else 0
    
    # Recall unigram (ROUGE-1 like)
    r1 = overlap_unigram / len(ref_tokens) if ref_tokens else 0
    
    # Bigrams
    ref_bigrams = set(zip(ref_tokens[:-1], ref_tokens[1:]))
    cand_bigrams = list(zip(cand_tokens[:-1], cand_tokens[1:]))
    overlap_bigram = sum(1 for bg in cand_bigrams if bg in ref_bigrams)
    p2 = overlap_bigram / len(cand_bigrams) if cand_bigrams else 0
    
    # Geometric mean approximation
    bleu_score = (p1 * (p2 if p2 > 0 else p1)) ** 0.5 * 100
    rouge1_score = (2 * p1 * r1 / (p1 + r1)) * 100 if (p1 + r1) > 0 else 0
    
    return {
        "BLEU-1": round(p1 * 100, 2),
        "BLEU-2": round(p2 * 100, 2),
        "Approx BLEU": round(bleu_score, 2),
        "ROUGE-1 (F1)": round(rouge1_score, 2)
    }

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("Settings & Controls")
    
    # Model Selection
    model_option = st.selectbox(
        "Deep Learning Model",
        options=[
            "Salesforce/codet5-base-multi-sum",
            "SEBIS/code_trans_t5_small_source_code_summarization_python"
        ],
        index=0,
        help="Pre-trained Transformer models fine-tuned on CodeSearchNet."
    )
    
    st.markdown("---")
    st.subheader("Inference Hyperparameters")
    beam_size = st.slider("Beam Search Size", min_value=1, max_value=8, value=4, step=1, 
                          help="Higher beam size yields higher quality generation at a slight speed trade-off.")
    max_output_tokens = st.slider("Max Summary Length", min_value=16, max_value=128, value=48, step=8,
                                  help="Maximum number of generated words/tokens.")
    ngram_penalty = st.slider("No-Repeat N-gram", min_value=1, max_value=4, value=2, step=1,
                              help="Prevents repetitive phrases in the summary.")
    
    st.markdown("---")
    st.markdown("### 🎓 Academic Info")
    st.markdown("**Subject:** Deep Learning")
    st.markdown("**Architecture:** Encoder-Decoder Transformer (T5 / CodeT5)")
    st.markdown("**Objective:** Code-to-Text Generation")

# ---------------------------------------------------------
# Main Interface
# ---------------------------------------------------------
st.markdown('<div class="main-title">Automated Source Code Summarizer & Comment Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">A Deep Learning-powered Code-to-Text system built with Pre-trained Transformers (CodeT5)</div>', unsafe_allow_html=True)

# Example selector
selected_preset = st.selectbox("📂 Load a Sample Snippet (or paste your own below):", list(PRESETS.keys()))

col1, col2 = st.columns([1, 1])

with col1:
    language = st.selectbox("Programming Language", ["Python", "JavaScript", "Java", "C++", "Go", "PHP"])

with col2:
    st.caption("Status & Hardware")
    import torch
    device_label = "CUDA GPU 🚀" if torch.cuda.is_available() else "CPU ⚙️"
    st.info(f"Target Execution Device: **{device_label}**")

# Code Input Area
default_code = PRESETS[selected_preset] if selected_preset != "Select an example..." else ""
code_input = st.text_area(
    "Source Code Input:",
    value=default_code,
    height=240,
    placeholder="def calculate_factorial(n):\n    if n <= 1:\n        return 1\n    return n * calculate_factorial(n - 1)"
)

# Action button
generate_btn = st.button("⚡ Generate Summary & Comments", type="primary", use_container_width=True)

# ---------------------------------------------------------
# Generation & Display
# ---------------------------------------------------------
if generate_btn:
    if not code_input.strip():
        st.warning("⚠️ Please paste or type some source code first!")
    else:
        with st.spinner("Loading Transformer model & generating neural summary..."):
            try:
                tokenizer, model = load_model_and_tokenizer(model_option)
                summary, latency, sample_tokens = generate_summary(
                    code_snippet=code_input,
                    tokenizer=tokenizer,
                    model=model,
                    num_beams=beam_size,
                    max_length=max_output_tokens,
                    no_repeat_ngram_size=ngram_penalty
                )
                
                # Store in session state for tabs
                st.session_state["summary"] = summary
                st.session_state["latency"] = latency
                st.session_state["code_input"] = code_input
                st.session_state["sample_tokens"] = sample_tokens
                st.session_state["language"] = language
                st.session_state["model_option"] = model_option
            except Exception as e:
                st.error(f"Error during model inference: {str(e)}")

# Display Results if available in session state
if "summary" in st.session_state:
    summary = st.session_state["summary"]
    latency = st.session_state["latency"]
    code_input = st.session_state["code_input"]
    sample_tokens = st.session_state["sample_tokens"]
    language = st.session_state["language"]
    model_option = st.session_state["model_option"]
    
    # Metrics Bar
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(label="Inference Latency", value=f"{latency:.2f} s")
    with m2:
        num_lines = len(code_input.strip().split("\n"))
        st.metric(label="Code Lines (LOC)", value=num_lines)
    with m3:
        token_count = len(code_input.split())
        st.metric(label="Approx Word Count", value=token_count)
    with m4:
        st.metric(label="Beam Search Beams", value=beam_size)

    st.markdown("---")

    # Tabs for presentation & grading
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Generated Summary & Comments",
        "🧠 Deep Learning Internals (Viva Demo)",
        "📊 Quantitative Evaluation (BLEU & ROUGE)",
        "ℹ️ Model & Dataset Architecture"
    ])

    with tab1:
        st.subheader("1. English Natural Language Summary")
        st.success(f"**Function Objective:** {summary.capitalize()}")
        
        st.subheader("2. Code with Injected Comments / Docstring")
        commented_code = inject_docstring(code_input, summary, language)
        st.code(commented_code, language=language.lower())
        
        st.download_button(
            label="💾 Download Commented Code",
            data=commented_code,
            file_name=f"commented_code.{'py' if language=='Python' else 'js' if language=='JavaScript' else 'txt'}",
            mime="text/plain"
        )

    with tab2:
        st.subheader("Transformer Subword Tokenization Breakdown")
        st.write(
            "Below are the actual subword tokens extracted by the Byte-Pair / SentencePiece tokenizer "
            "before feeding into the Transformer Encoder:"
        )
        
        # Display tokens nicely
        token_html = " ".join([f'<span class="badge" style="margin: 2px;">{repr(t)[1:-1]}</span>' for t in sample_tokens])
        st.markdown(token_html, unsafe_allow_html=True)
        
        st.markdown("### Encoder-Decoder Flow")
        st.markdown("""
        ```
        Source Code ──> [Tokenizer] ──> [Input IDs + Positional Encodings]
                                                  │
                                                  ▼
                                      [CodeT5 Encoder Layers]
                                      (Multi-Head Self-Attention)
                                                  │
                                                  ▼
        Generated Text <── [Decoder] <── [Cross-Attention]
        ```
        """)
        st.info("💡 **Viva Tip:** Explain to the examiner that the **Encoder** learns representations of code structure (AST, identifiers, loops), while the **Decoder** autoregressively generates the natural language description token by token using **Cross-Attention**.")

    with tab3:
        st.subheader("Quantitative Metric Verification (BLEU / ROUGE)")
        st.write(
            "Enter a Ground Truth (Reference) description or docstring to evaluate the model's accuracy on this code:"
        )
        
        default_ref = "Calculates the target element position using binary search." if "binary_search" in code_input else ""
        ref_input = st.text_input("Ground Truth Reference Summary:", value=default_ref, placeholder="Type reference docstring...")
        
        if ref_input.strip():
            metrics = compute_simple_bleu(ref_input, summary)
            c1, c2, c3 = st.columns(3)
            c1.metric("BLEU-1 (Unigram Precision)", f"{metrics['BLEU-1']}%")
            c2.metric("BLEU-2 (Bigram Precision)", f"{metrics['BLEU-2']}%")
            c3.metric("ROUGE-1 (F1 Overlap)", f"{metrics['ROUGE-1 (F1)']}%")
            
            st.caption("Standard automated metrics used in Code-to-Text research papers (CodeSearchNet Benchmark).")
        else:
            st.caption("Enter a reference docstring above to compute BLEU and ROUGE scores.")

    with tab4:
        st.subheader("Model Information")
        st.markdown(f"""
        - **Model Selected:** `{model_option}`
        - **Original Training Dataset:** CodeSearchNet (over 2 million code-comment pairs across Python, Java, JS, Go, Ruby, PHP).
        - **Primary Task:** Code-to-Text (Source Code Summarization).
        - **Architecture:** T5 (Text-to-Text Transfer Transformer) pre-trained with identifier-tagging and masked language modeling objectives.
        """)
