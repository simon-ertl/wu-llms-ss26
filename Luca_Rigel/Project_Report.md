# ==============================================================================
# PROJECT REPORT: Austrian Tax Law LLM
# ==============================================================================

## 1. Models and Baseline Configuration

### Initial Approach (Failure Mode)
Initially, the `google/flan-t5-small` architecture was deployed across all models. However, the model's parameter count proved insufficient for the complexity of the legal domain. It exhibited consistent hallucination and produced incoherent mixtures of English and German text.

### Final Stable Configurations
To address these limitations, the architecture was upgraded to the more capable `Qwen2.5` model family, which yielded high-quality and contextually accurate German language generation:
- **Model 1 (Zero-Shot Inference):** `Qwen/Qwen2.5-3B-Instruct`
- **Model 2 (Fine-Tuning):** `Qwen/Qwen2.5-1.5B-Instruct`
- **Model 3 (RAG):** `Qwen/Qwen2.5-3B-Instruct` (Generator) + `MiniLM-L12` (Retriever)

### Hyper-parameters & Setup (Model 1 Baseline):
- **Model Size:** 3 Billion parameters.
- **Sampling Approach:** Stochastic generation with a low temperature (`T=0.1`) and `do_sample=True` to allow slight variability while maintaining high precision. The `max_new_tokens` parameter was set to 250.
- **Pre-training Data:** Qwen2.5 is pre-trained on a massive multilingual corpus (up to 18T tokens) encompassing coding, mathematics, and high-quality multilingual texts. This extensive training accounts for its native fluency in German tax logic compared to FLAN-T5.

---

## 2. Fine-Tuned Model (Model 2)

### Data
Due to the absence of a pre-existing instruction dataset for Austrian Tax Law, an **automated self-instruct pipeline** was developed to generate training data:
- **TF-IDF Retrieval:** For each of the 644 queries, the most relevant PDF paragraph was retrieved using a TF-IDF vectorizer (max_features=10,000, filtered for common German stop words). 
- **Training Pairs:** 150 high-quality question-context pairs were synthesized. The model was trained to answer questions strictly based on the provided context, utilizing a Causal LM objective and the ChatML format.

### Fine-Tuning Strategy
- **Method:** PEFT / LoRA (Low-Rank Adaptation) was employed to fine-tune the 1.5B parameter model within the VRAM constraints of a Kaggle T4 GPU (15GB).
- **Quantization:** 4-bit NormalFloat (NF4) quantization was deployed via BitsAndBytes.
- **Hyper-parameters:**
  - Rank (r) = 8, Alpha = 16, Dropout = 0.05
  - Target Modules: `q_proj`, `v_proj`
  - Max Steps = 50, Batch Size = 2, Gradient Accumulation = 4
  - Optimizer: AdamW, Learning Rate = 2e-4, FP16 precision.

---

## 3. RAG-Based Model (Model 3)

### Retrieval Model
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (a dense neural vector model highly optimized for multilingual semantic search).

### Document Indexing & Preprocessing (Chunking)
- The official Tax Law PDFs were parsed utilizing `pypdf`.
- **Chunking:** Pages were extracted, stripped of noise (e.g., footers, email addresses), and split by double line breaks (`\n\n`) into distinct semantic paragraphs. Excessively small fragments (< 80 characters) were discarded.
- **Embedding:** All cleaned paragraphs were encoded into dense vectors using L2 normalization to facilitate rapid Cosine Similarity matching via dot product calculations.

### Input Passages (Top-K)
- For every query, the generator was provided with the **top 3 (k=3)** most semantically similar paragraphs. The maximum length of the concatenated retrieved context was capped at 1,000 characters to prevent context-window overflow.

---

## 4. Evaluation & Performance Metrics

Since a dedicated, finalized shared annotation task dataset was unavailable, the full official `Austrian Tax Law Dataset`—comprising roughly 680 queries paired with their `correct_answer`—was utilized as the Ground Truth baseline. 

An automated evaluation pipeline (`evaluate_models.py`) iterated over the entire dataset, comparing the LLM-generated output against the provided human-curated ground truth to compute BLEU (SacreBLEU) and ROUGE-L metrics.

### Main Result Table
| Model Strategy | Base Model | BLEU Score | ROUGE-L (F1) | BERTScore | Execution Setup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model 1 (Zero-Shot)** | `Qwen2.5-3B-Instruct` | **3.13** | **0.1492** | **0.6975** | Prompt Only |
| **Model 2 (Fine-Tuned)** | `Qwen2.5-1.5B-Instruct` | 1.31 | 0.0801 | 0.6537 | LoRA (r=8) |
| **Model 3 (RAG)** | `Qwen2.5-3B-Instruct` | 2.72 | 0.1264 | 0.6792 | Top-3 Semantic |

<br>

<div align="center">
  <img src="./results/model_performance_chart.png" alt="Model Performance Comparison" width="800">
</div>

<br>

**Conclusion:** Generative LLMs naturally paraphrase information rather than outputting exact memorized string sequences. Consequently, n-gram based metrics (BLEU and ROUGE-L) severely penalized all models. However, the integration of **BERTScore**—which evaluates semantic distance via high-dimensional vector embeddings—provided a more representative assessment: all models achieved a semantic similarity of roughly ~0.65 to 0.70 compared to the human ground truth. Ultimately, the robust baseline performance of the `Qwen2.5-3B` parameter model (Model 1) marginally outperformed the others, with the RAG Model (Model 3) following closely by dynamically leveraging fetched legal evidence.

---

## 5. Error Analysis and Limitations

Throughout the development and testing phases, several distinct failure modes were identified and addressed across the models:

1. **Model 1 (Hallucination without grounding):**
   - *Issue:* Pure zero-shot models often confidently asserted generic German tax logic (e.g., German BGB/EStG) instead of Austrian specificities (öKStG/öEStG) due to the overwhelming presence of federal German data in their pre-training corpus.
2. **Model 2 (Metadata Leakage & Infinite Repetition Loops):**
   - *Issue:* During initial evaluations, the fine-tuned model inadvertently generated string artifacts such as `"Lizenziert für: Viktoria.Schwab@aau.at"`, indicating memorization of noisy PDF footers. 
   - *Fix:* Implementing strict regex filtering prior to tokenization resolved the artifact leakage. Furthermore, the early architectures (`flan-t5-small`) frequently fell into infinite repetition loops during inference, causing extensive execution times and corrupted outputs. By upgrading to `Qwen2.5` and enforcing strict generation penalties (`no_repeat_ngram_size=3`, `repetition_penalty=1.2`), generative looping was entirely eliminated, resulting in efficient execution and highly accurate text generation.
3. **Model 3 (Retrieval Misses):**
   - *Issue:* When queries utilized highly abstract phrasing that was not semantically aligned with the dense legal text, the MiniLM retriever fetched irrelevant paragraphs. Confronted with irrelevant context, the generator occasionally forced a hallucination.
   - *Fix:* Implementing a strict system prompt ("NUTZE AUSSCHLIESSLICH den bereitgestellten Kontext") substantially reduced RAG-induced hallucinations, safely converting retrieval misses into appropriate "I don't know" abstentions.
