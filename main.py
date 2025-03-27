from reasoning_engine import SymbolicReasoner
from embedding_engine import EmbeddingEngine
from reasoning_engine import visualize_reasoning_chain

# Load modules
reasoner = SymbolicReasoner("data/scientific_rules.txt")
embedder = EmbeddingEngine()

# Ask user for observation
observation = input("Enter a scientific observation or symptom: ").strip()

# Step 1: Neural - Get related concepts + facts
print("\n🧠 Neural System Output")

top_concepts = embedder.get_related_concepts(observation, top_k=5)
top_facts = embedder.get_related_facts(observation, top_k=3)

print("Top related concepts (from embeddings):")
for concept in top_concepts:
    print(f" - {concept}")

print("\nTop related facts:")
for fact in top_facts:
    print(f" - {fact}")

# Step 2: Symbolic Reasoning with Abductive Scoring
# Step 2: Symbolic Reasoning with Abductive Scoring
print("\n🤖 Abductive Reasoning: Inference to Best Explanation")

best_chain, best_score, all_chains = reasoner.select_best_explanation(
    top_concepts, top_facts, observation, embedder
)

if best_chain:
    print(f"\n✅ Best Explanation (score = {best_score}):")
    for step in best_chain:
        print(f"  {step[0]} => {step[1]}")
else:
    print("No valid symbolic explanation found.")

# Step 3: Natural Language Explanation
from reasoning_engine import explain_chain_naturally

print("\n📝 Natural Language Explanation:")
print(explain_chain_naturally(best_chain))
visualize_reasoning_chain(best_chain, title="Abductive Reasoning Path")

