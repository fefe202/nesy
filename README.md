# NeSy Course Projects

This repository contains the projects for the **Neurosymbolic AI** course, divided into two main modules.

## Project Structure

- **`Foundations/`**: Contains the Neural Symbolic project focused on semi-supervised learning and semantic loss implementation using PyTorch.
- **`LLM_Lab/`**: Contains the **Business Intelligence Co-pilot**, a RAG-based application built with Streamlit and LLMs for market analysis.

---

## Requirements & Installation

### Global Prerequisites
- **Python 3.10+**
- **Ollama** (Required for *LLM_Lab*)

### 1. Foundations (Neural Symbolic)

This module requires a standard Data Science/Deep Learning environment.

**Dependencies:**
```bash
pip install torch torchvision numpy matplotlib tqdm jupyter
```

**Usage:**
1. Import the notebook `notebooke8ba85730c (1).ipynb` into [Kaggle](https://www.kaggle.com/).
2. In the notebook editor, go to **Settings** (right sidebar).
3. Under **Accelerator**, select **GPU P100**.
4. Run the notebook.

### 2. LLM_Lab (Business Intelligence Co-pilot)

This acts as a web application utilizing Local LLMs (via Ollama) and Vector Databases (ChromaDB).

**Dependencies:**
```bash
pip install streamlit pandas chromadb sentence-transformers langchain-ollama langchain duckduckgo-search streamlit-elements
```

**Ollama Setup:**
You need to have [Ollama](https://ollama.com/) installed and running. Pull the required models before running the app:
```bash
ollama pull gemma3:4b
ollama pull llama3.2
```
*(Note: Ensure Ollama is running in the background, typically on port 11434)*

**Usage:**
Run the Streamlit application from the `LLM_Lab` directory:
```bash
cd LLM_Lab
streamlit run project_llm_lab.py
```

## Folder Overview

### `/Foundations`
- `notebooke...ipynb`: The main notebook implementing a **Sorting Network** with **Semantic Loss** regularization. It demonstrates how to enforce logical constraints (permutations) on a neural network output in a semi-supervised setting using MNIST data.

### `/LLM_Lab`
- `project_llm_lab.py`: Main Streamlit application file.
- `Business Co-pilot Presentation.pdf` / `Report_LLM.pdf`: Documentation and slides for the project.
- `imprese_attive_2021.csv`: Dataset used for the RAG knowledge base.

