# 📚 Academic Research Paper Assistant

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.45.1-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated RAG (Retrieval-Augmented Generation) application designed to help researchers analyze, compare, and extract insights from academic papers using LangChain, FAISS, and Google's Gemini API.

## 🌟 Features

- **Multi-Paper Analysis**: Upload and process multiple research papers simultaneously.
- **Intelligent Chunking**: Preserves academic paper sections and context during text splitting.
- **RAG Pipeline**: Uses TF-IDF vector search for precise answers.
- **Source Attribution**: Displays the paper and section for each response.
- **Research-Specific Queries**: Supports methodology comparison, result analysis, and citation networks.
- **Modern UI**: Clean, responsive Streamlit interface with a single, eye-friendly color scheme.
- **CPU Optimized**: Efficient FAISS indexing and chunking for deployment on Streamlit Cloud.

## 🚀 Live Demo

[https://joy-academic-research-assistant.streamlit.app]

## 🛠️ Prerequisites

- Python 3.12
- Google Gemini API key (obtain from [Google AI](https://ai.google.dev/))

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/research-paper-assistant.git
   cd research-paper-assistant
   ```
