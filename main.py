from reasoning_engine import SymbolicReasoner
from embedding_engine import EmbeddingEngine

# Load modules
reasoner = SymbolicReasoner("data/scientific_rules.txt")
embedder = EmbeddingEngine()

# User input
observation = input("Enter a scientific observation (e.g., memory_loss): ").strip()

# Step 1: Symbolic reasoning
print("\n--- Symbolic Hypotheses ---")
chains = reasoner.explain(observation)
if not chains:
    print("No symbolic hypothesis found.")
else:
    for i, chain in enumerate(chains):
        print(f"\nHypothesis {i+1}:")
        for step in chain:
            print(f"{step[0]} => {step[1]}")

# Step 2: Neural semantic similarity
print("\n--- Neural Similar Concepts ---")
concept_list = [
    "disruption of pnn", "low neurosteroid levels", "microglial activation", 
    "aromatase suppression", "forgetfulness", "impaired learning", "aging"
]
similar = embedder.get_similar_concepts(observation, concept_list)
for concept, score in similar:
    print(f"{concept} (similarity: {score:.2f})")
