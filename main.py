from reasoning_engine import SymbolicReasoner
from embedding_engine import EmbeddingEngine

# Load modules
reasoner = SymbolicReasoner("data/scientific_rules.txt")
embedder = EmbeddingEngine()

# Ask user for mode
mode = input("Choose mode: (1) Explain a symptom (backward), (2) Predict outcomes (forward): ").strip()

# Get concept
concept = input("Enter a scientific concept (e.g., memory_loss or low neurosteroid levels): ").strip()

# Step 1: Symbolic Reasoning
if mode == "1":
    print("\n--- EXPLAIN: Symbolic Hypotheses ---")
    chains = reasoner.explain(concept)
elif mode == "2":
    print("\n--- PREDICT: Symbolic Outcomes ---")
    chains = reasoner.predict(concept)
else:
    print("Invalid mode.")
    chains = []

if not chains:
    print("No symbolic paths found.")
else:
    for i, chain in enumerate(chains):
        print(f"\nChain {i+1}:")
        for step in chain:
            print(f"{step[0]} => {step[1]}")

# Step 2: Neural similarity
print("\n--- Neural Similar Concepts ---")
concept_list = [
    "disruption of pnn", "low neurosteroid levels", "microglial activation", 
    "aromatase suppression", "forgetfulness", "impaired learning", "aging"
]
similar = embedder.get_similar_concepts(concept, top_k=5)
for concept, score in similar:
    print(f"{concept} (similarity: {score:.2f})")