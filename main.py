# main.py

from pubmed_query import fetch_pubmed_data  # Import PubMed query function
from reasoning_engine import SymbolicReasoner
from embedding_engine import EmbeddingEngine
from reasoning_engine import visualize_reasoning_chain
from reasoning_engine import SymbolicReasoner, visualize_reasoning_chain, explain_chain_naturally
import urllib.parse

# Load modules
reasoner = SymbolicReasoner("data/scientific_rules.txt")
embedder = EmbeddingEngine()

# Ask user for observation (now you might want to query PubMed based on observation)
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

# Step 2: Query PubMed for relevant papers based on observation
print("\n📚 Fetching Related Scientific Papers from PubMed")

# In your main.py, before calling fetch_pubmed_data():
observation = input("Enter a scientific observation or symptom: ").strip()
encoded_observation = urllib.parse.quote(observation)
arxiv_papers = fetch_pubmed_data(encoded_observation, max_results=3)

pubmed_papers = fetch_pubmed_data(observation, max_results=3)  # Example of querying PubMed

for paper in pubmed_papers:
    print(f"Title: {paper['title']}")
    print(f"Summary: {paper['summary']}")
    print(f"Authors: {', '.join(paper['authors'])}")
    print(f"Published: {paper['published']}")
    print(f"Link: {paper['link']}")
    print("="*50)

arxiv_papers = fetch_pubmed_data(observation, max_results=3)
if not arxiv_papers:
    print("No papers found for this query.")
else:
    for paper in arxiv_papers:
        print(f"Title: {paper['title']}")
        print(f"Summary: {paper['summary']}")
        print(f"Authors: {', '.join(paper['authors'])}")
        print(f"Published: {paper['published']}")
        print(f"Link: {paper['link']}")
        print("="*50)

# Step 3: Symbolic Reasoning with Abductive Scoring
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

# Step 4: Natural Language Explanation
print("\n📝 Natural Language Explanation:")
print(explain_chain_naturally(best_chain))

# Visualize reasoning chain
visualize_reasoning_chain(best_chain, title="Abductive Reasoning Path")
