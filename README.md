# 🌍 Global Language Detector AI
An optimized Machine Learning pipeline utilizing statistical Natural Language Processing (NLP) to detect textual languages across multiple script systems in real-time.

---

## 🔬 Project Overview & Abstract
This repository contains a production-ready system engineered to address cross-lingual text classification. Utilizing a lightweight probabilistic approach, the architecture balances computational latency with classification accuracy, making it highly suitable for enterprise content routing, cross-lingual scraping workflows, and multilingual user-interfaces.

### Key Focus Areas:
* **Academic/Research:** Evaluates n-gram tokenization effectiveness on low-resource scripts (e.g., Urdu, Persian, Pushto) vs. Latin/Germanic languages.
* **Professional Engineering:** Packaged inside an automated Scikit-Learn `Pipeline` framework to prevent data leakage during train-test splitting.
* **Production Deployment:** Bundled with a high-performance Streamlit UI configured for cloud environments.

---

## 🧬 Methodology & Architecture

The classification subsystem relies on a combination of character/word tokenization and probabilistic generative modeling:

1. **Preprocessing Layer:** Handles PyArrow memory references and indices, standardizes dynamic text arrays into native Python strings, and isolates null fields.
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

> ℹ️ *Note: Actual exact validation accuracy is dynamically rendered in the Streamlit Sidebar directly from your custom `language.csv` dataset configuration during initialization.*

---

## 💻 Tech Stack & Frameworks
* **Language:** Python
* **ML Infrastructure:** Scikit-Learn
* **Data Arrays:** Pandas, NumPy
* **Interface & Cloud Deployment:** Streamlit Web Server

