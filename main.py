# main.py

import urllib.parse
from pubmed_query import fetch_pubmed_data
from reasoning_engine import SymbolicReasoner, visualize_reasoning_chain, explain_chain_naturally
from embedding_engine import EmbeddingEngine
from observation_extractor import extract_observations
from dataset_manager import update_facts

def get_user_choice(options, prompt):
    """
    Display options and return the user's chosen option.
    """
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    while True:
        try:
            choice = int(input(prompt))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            else:
                print(f"Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Define a structured set of neuroscience categories and their subfields.
categories = {
    "Cognitive Neuroscience": ["attention", "memory", "decision-making", "executive function"],
    "Behavioral Neuroscience": ["reward", "motivation", "addiction", "emotional regulation"],
    "Molecular Neuroscience": ["synaptic plasticity", "neurotransmitters", "gene expression", "cell signaling"],
    "Clinical Neuroscience": ["neurodegeneration", "stroke", "epilepsy", "psychiatry"]
}

# Step 1: Let the user choose a neuroscience category.
print("Select a Neuroscience Category from the following options:")
category = get_user_choice(list(categories.keys()), "Enter the number corresponding to your category: ")

# Step 2: Let the user choose a subfield from the selected category.
print(f"\nSelect a subfield related to {category}:")
subfield = get_user_choice(categories[category], "Enter the number corresponding to your subfield: ")

# Step 3: Instruct the user to enter a set of keywords (not a full question) that are likely to work in PubMed.
print("\nNow, please enter a set of search keywords (separated by spaces) to refine your query.")
print("For example, if you selected 'Cognitive Neuroscience' and 'attention', you might enter:")
print("  perineuronal nets synaptic plasticity")
keywords = input("Enter your search keywords: ").strip()

# Construct the final query string using Boolean operators.
# This forms a query like: "Cognitive Neuroscience AND attention AND perineuronal nets AND synaptic plasticity"
query_components = [category, subfield] + keywords.split()
final_query = " AND ".join(query_components)
encoded_query = urllib.parse.quote(final_query)
print(f"\nYour final PubMed query is: '{final_query}'")

# Initialize modules.
reasoner = SymbolicReasoner("data/scientific_rules.txt")
embedder = EmbeddingEngine()

# Step 4: Fetch Papers from PubMed with a limit on max results.
MAX_PAPERS = 5  # Adjust as needed.
print(f"\nFetching up to {MAX_PAPERS} related scientific papers from PubMed for query: '{final_query}' ...")
papers = fetch_pubmed_data(encoded_query, max_results=MAX_PAPERS)

if not papers:
    print("No papers found for this query. Please try refining your keywords.")
else:
    for i, paper in enumerate(papers, 1):
        print(f"\nPaper #{i}")
        print(f"Title: {paper['title']}")
        if paper['abstract']:
            # Truncate abstract for readability.
            print(f"Abstract: {paper['abstract'][:500]}{'...' if len(paper['abstract']) > 500 else ''}")
        else:
            print("No abstract available.")

# Step 5: Extract Observations from Papers (limit total extracted observations).
MAX_OBSERVATIONS = 5
extracted_observations = []
obs_count = 0

print("\nExtracting Observations from Papers...")
for paper in papers:
    if paper.get("abstract"):
        if obs_count >= MAX_OBSERVATIONS:
            break  # Stop extraction once the limit is reached.
        obs = extract_observations(paper["abstract"])
        extracted_observations.append(obs)
        obs_count += 1
        print(f"Extracted Observation: {obs}")
    else:
        print("Skipping paper due to lack of abstract.")

# Fallback: if no observations are extracted, use the chosen subfield.
combined_observation = " ".join(extracted_observations) if extracted_observations else subfield

# Step 6: Neural Processing - Get related concepts and facts.
print("\nNeural System Processing...")
top_concepts = embedder.get_related_concepts(combined_observation, top_k=5)
top_facts = embedder.get_related_facts(combined_observation, top_k=3)

print("Top Related Concepts:")
for concept in top_concepts:
    print(f" - {concept}")

print("\nTop Related Facts:")
for fact in top_facts:
    print(f" - {fact}")

# Optionally, update the dynamic facts dataset with the extracted observations.
for obs in extracted_observations:
    update_facts(obs)

# Step 7: Symbolic Reasoning with Abductive Inference.
print("\nPerforming Abductive Reasoning to Generate Explanations...")
extended_known_facts = top_facts + extracted_observations
best_chain, best_score, all_chains = reasoner.select_best_explanation(
    top_concepts, extended_known_facts, combined_observation, embedder
)

if best_chain:
    print(f"\nBest Explanation (score = {best_score}):")
    for step in best_chain:
        print(f"  {step[0]} => {step[1]}")
else:
    print("No valid symbolic explanation found.")

# Step 8: Generate Natural Language Explanation.
print("\nNatural Language Explanation:")
nl_explanation = explain_chain_naturally(best_chain)
print(nl_explanation)

# Step 10: Visualize the Reasoning Chain.
visualize_reasoning_chain(best_chain, title="Abductive Reasoning Path")
