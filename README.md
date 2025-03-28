# 🧠 Abductive Neuroscientist

A neurosymbolic system that simulates **abductive reasoning** over neuroscience literature. It integrates neural embeddings, symbolic logic, and scientific rules to explain observations with natural language and visual reasoning chains.

This tool allows researchers and students to input neuroscience keywords, fetch relevant PubMed papers, extract key observations, and receive symbolic explanations grounded in real scientific knowledge.

---

## 🧩 Features

- **Neural Embedding Engine**: Embeds neuroscience concepts and facts for semantic similarity matching.
- **Symbolic Reasoning Engine**: Traverses scientific rules to construct abductive reasoning chains.
- **Natural Language Explanation**: Converts logic chains into readable, causal explanations.
- **Visual Graphs**: Displays reasoning chains as graphs using `NetworkX` and `matplotlib`.
- **Live PubMed Integration**: Searches and parses abstracts via NCBI's PubMed API.
- **Observation Extraction**: Uses a BART summarizer to derive scientific observations from abstracts.
- **Dynamic Dataset Updates**: Adds new observations to your knowledge base for iterative learning.

---

## 📁 Project Structure

```bash
abductive_scientist/
│
├── main.py                      # Main CLI pipeline for reasoning workflow
├── reasoning_engine.py          # SymbolicReasoner class and visualization
├── embedding_engine.py          # EmbeddingEngine using SentenceTransformers
├── pubmed_query.py              # Query PubMed API and parse XML responses
├── observation_extractor.py     # Summarize key observations using BART
├── dataset_manager.py           # Append new facts to dataset
│
├── data/
│   ├── concepts.txt             # Concepts and their definitions
│   ├── facts.txt                # Scientific fact statements
│   ├── scientific_rules.txt     # Symbolic rules in "A => B" format
│
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/abductive_scientist.git
cd abductive_scientist
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline

```bash
python main.py
```

You'll be guided through a multi-step CLI:
- Select a neuroscience category and subfield.
- Enter search keywords.
- Fetch papers and extract observations.
- Generate reasoning chains and explanations.
- Visualize the result.

---

## 🧠 Example Output

```
Best Explanation (score = 6.73):
  chronic_stress => glucocorticoid_excess
  glucocorticoid_excess => hippocampal_atrophy
  hippocampal_atrophy => memory_loss

Natural Language Explanation:
Because chronic stress, it may lead to glucocorticoid excess.
Because glucocorticoid excess, it may lead to hippocampal atrophy.
Because hippocampal atrophy, it may lead to memory loss.
Therefore, the observed issue may ultimately be due to chronic stress.
```

---

## 📚 Datasets

All domain knowledge is stored in the `data/` folder:
- `concepts.txt`: Concept name and definition (e.g., `memory_loss: ...`)
- `facts.txt`: One-liner factual statements.
- `scientific_rules.txt`: Symbolic implications (e.g., `A => B`)

You can extend or refine these files as your dataset grows.

---

## 🛠 Requirements

- `sentence-transformers`
- `transformers`
- `networkx`
- `matplotlib`
- `requests`

Pretrained models used:
- `facebook/bart-large-cnn` (for summarization)
- `all-MiniLM-L6-v2` (for embeddings)

---

## 🧪 Development Notes

- `SymbolicReasoner` includes a scoring function that combines symbolic length, fact overlap, and semantic similarity.
- Explanations are ranked using cosine similarity between the user’s query and generated chain summaries.
- Reasoning chains can be visualized as graphs for better interpretability.

---

## 📖 Citation & Inspiration

This project draws from work in **neurosymbolic AI**, **scientific discovery**, and **computational neurocognition**.

If you use this tool in academic research, consider citing or linking back to the repository.

---

## 🧬 Future Work

- Automatic PDF-to-rule extraction.
- Integration with PubMed Central full-text.
- GUI or web interface.
- Hypothesis generation module.
- Export to LaTeX for academic reporting.

---

## 🧑‍💻 Authors

Built with care by [Jiya Manchanda](https://github.com/jiya-manchanda). Contributions welcome!
