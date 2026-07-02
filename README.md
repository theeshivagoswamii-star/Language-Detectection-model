# 🌍 Global Language Detector AI
An optimized Machine Learning pipeline utilizing statistical Natural Language Processing (NLP) to detect textual data across 20+ global languages in real-time.

---

## 🔬 About the Project
This repository contains a production-ready system engineered to address cross-lingual text classification. Utilizing a lightweight probabilistic approach, the architecture balances computational latency with high classification accuracy, making it highly suitable for enterprise content routing, cross-lingual scraping workflows, and multilingual user interfaces.

The core core focus of this system is its ability to handle **20+ global languages** natively, breaking down linguistic barriers by supporting multiple distinct script domains on a singular vector space model.

### 🌐 Supported Linguistic Domains & Scripts
The architecture is trained and optimized to process diverse global language families seamlessly:
* **Latin / Germanic / Romance Scripts:** English, Spanish, French, German, Portuguese, Italian, Dutch, Romanian, Estonian, Swedish.
* **Devanagari / Indo-Aryan Scripts:** Hindi, Marathi.
* **Perso-Arabic Scripts:** Urdu, Persian, Pushto.
* **East Asian & Southeast Asian Scripts:** Chinese, Korean, Japanese, Thai.
* **Cyrillic Script:** Russian.

### 🚀 Key Focus Areas
* **Multi-Script Vectorization:** Designed to seamlessly process mixed-character vocabularies without losing structural or syntactic context.
* **Academic/Research Benchmark:** Evaluates n-gram tokenization effectiveness on low-resource and high-inflection languages compared to classic European structures.
* **Enterprise Integration:** Fully wrapped inside a Scikit-Learn `Pipeline` framework to completely isolate inference logic and prevent data leakage.

---

## 🧬 Methodology & Architecture

The classification subsystem relies on a combination of character/word tokenization and probabilistic generative modeling:

1. **Data Cleaning Layer:** Standardizes structural dynamic text arrays into native Python strings, filters null parameters, and prevents PyArrow indexing errors.
2. **Feature Extraction:** Implements an `ngram_range=(1, 2)` `CountVectorizer` matrix to analyze both isolated word vocabularies and contextual contiguous phrases.
3. **Inference Engine:** Features a `MultinomialNB` classifier calibrated with an additive Laplace smoothing parameter ($\alpha = 0.1$) to account for out-of-vocabulary (OOV) tokens during live validation.

---

## 📊 Performance Metrics

The model achieves deterministic consistency across evaluation test sets:

| Metric | Evaluation Value |
| :--- | :--- |
| **Algorithm** | Multinomial Naive Bayes Classifier |
| **Vector Space Model** | Document-Term Matrix (Token Counts) |
| **Test Split Ratio** | 20% Holdout Validation Set |
| **Smoothing Configuration** | $\alpha = 0.1$ (Laplace) |
| **Language Vocabulary Size** | Scaled across 20+ Global Alphabets |

> ℹ️ *Note: The exact validation accuracy is dynamically rendered in the Streamlit Sidebar directly from your custom local `language.csv` dataset configuration during initialization.*

---

## 💻 Tech Stack & Frameworks
* **Language:** Python
* **ML Infrastructure:** Scikit-Learn
* **Data Arrays:** Pandas, NumPy
* **Interface & Cloud Deployment:** Streamlit Web Server

---

## 🚀 Installation & Local Execution

### 1. Clone the Space
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
