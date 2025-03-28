# main.py

import urllib.parse
import logging

from pubmed_query import fetch_pubmed_data
from reasoning_engine import SymbolicReasoner, visualize_reasoning_chain, explain_chain_naturally
from embedding_engine import EmbeddingEngine
from observation_extractor import extract_observations
from dataset_manager import update_facts

# Configure logging to output messages with timestamps and log levels.
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def get_user_choice(options, prompt):
    """
    Display a list of options and prompt the user to select one.

    Parameters:
        options (list): A list of string options to display.
        prompt (str): The prompt message to show for input.

    Returns:
        str: The option selected by the user.
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

def main():
    """
    Main function orchestrating the AI pipeline:
    
    1. User selects a neuroscience category and a corresponding subfield.
    2. User provides search keywords to refine a PubMed query.
    3. Articles are fetched from PubMed based on the query.
    4. Observations are extracted from the articles' abstracts.
    5. Neural processing retrieves related concepts and facts.
    6. The facts dataset is updated with newly extracted observations.
    7. Symbolic reasoning generates and scores explanation chains.
    8. The best explanation is presented in both structured and natural language.
    9. The reasoning chain is visualized as a directed graph.
    """
    try:
        # Define a structured set of neuroscience categories and their subfields.
        categories = {
            "Cognitive Neuroscience": ["attention", "memory", "decision-making", "executive function"],
            "Behavioral Neuroscience": ["reward", "motivation", "addiction", "emotional regulation"],
            "Molecular Neuroscience": ["synaptic plasticity", "neurotransmitters", "gene expression", "cell signaling"],
            "Clinical Neuroscience": ["neurodegeneration", "stroke", "epilepsy", "psychiatry"]
        }

        # Step 1: User selects a neuroscience category.
        print("Select a Neuroscience Category from the following options:")
        category = get_user_choice(list(categories.keys()), "Enter the number corresponding to your category: ")

        # Step 2: User selects a subfield within the chosen category.
        print(f"\nSelect a subfield related to {category}:")
        subfield = get_user_choice(categories[category], "Enter the number corresponding to your subfield: ")

        # Step 3: User provides search keywords to refine the PubMed query.
        print("\nNow, please enter a set of search keywords (separated by spaces) to refine your query.")
        print("For example, if you selected 'Cognitive Neuroscience' and 'attention', you might enter:")
        print("  perineuronal nets synaptic plasticity")
        keywords = input("Enter your search keywords: ").strip()

        # Construct the final query string.
        query_components = [category, subfield] + keywords.split()
        final_query = " AND ".join(query_components)
        encoded_query = urllib.parse.quote(final_query)
        print(f"\nYour final PubMed query is: '{final_query}'")

        # Step 4: Initialize the necessary modules.
        reasoner = SymbolicReasoner("data/scientific_rules.txt")
        embedder = EmbeddingEngine()

        # Step 5: Fetch papers from PubMed.
        MAX_PAPERS = 5
        logging.info("Fetching up to %d related scientific papers from PubMed for query: '%s'", MAX_PAPERS, final_query)
        papers = fetch_pubmed_data(encoded_query, max_results=MAX_PAPERS)
        if not papers:
            logging.error("No papers found for this query. Please try refining your keywords.")
            return
        else:
            for i, paper in enumerate(papers, 1):
                print(f"\nPaper #{i}")
                print(f"Title: {paper['title']}")
                if paper['abstract']:
                    # Truncate abstract for readability.
                    print(f"Abstract: {paper['abstract'][:500]}{'...' if len(paper['abstract']) > 500 else ''}")
                else:
                    print("No abstract available.")

        # Step 6: Extract observations from the fetched papers.
        MAX_OBSERVATIONS = 5
        extracted_observations = []
        obs_count = 0
        logging.info("Extracting observations from papers...")
        for paper in papers:
            if paper.get("abstract"):
                if obs_count >= MAX_OBSERVATIONS:
                    break  # Limit the number of observations.
                obs = extract_observations(paper["abstract"])
                extracted_observations.append(obs)
                obs_count += 1
                print(f"Extracted Observation: {obs}")
            else:
                logging.warning("Skipping paper due to lack of abstract.")

        # Fallback: if no observations were extracted, use the chosen subfield.
        combined_observation = " ".join(extracted_observations) if extracted_observations else subfield

        # Step 7: Retrieve related concepts and facts using the embedding engine.
        logging.info("Performing neural processing to retrieve related concepts and facts...")
        top_concepts = embedder.get_related_concepts(combined_observation, top_k=5)
        top_facts = embedder.get_related_facts(combined_observation, top_k=3)

        print("Top Related Concepts:")
        for concept in top_concepts:
            print(f" - {concept}")

        print("\nTop Related Facts:")
        for fact in top_facts:
            print(f" - {fact}")

        # Step 8: Update the dataset with the newly extracted observations.
        logging.info("Updating fact dataset with extracted observations...")
        for obs in extracted_observations:
            update_facts(obs)

        # Step 9: Perform symbolic reasoning to generate explanation chains.
        logging.info("Performing symbolic reasoning to generate explanation chains...")
        extended_known_facts = top_facts + extracted_observations
        best_chain, best_score, all_chains = reasoner.select_best_explanation(
            top_concepts, extended_known_facts, combined_observation, embedder
        )

        # Display the best explanation if available.
        if best_chain:
            print(f"\nBest Explanation (score = {best_score}):")
            for step in best_chain:
                print(f"  {step[0]} => {step[1]}")
        else:
            logging.error("No valid symbolic explanation found.")
            print("No valid symbolic explanation found.")

        # Step 10: Generate a natural language explanation.
        logging.info("Generating natural language explanation from the best chain...")
        nl_explanation = explain_chain_naturally(best_chain)
        print("\nNatural Language Explanation:")
        print(nl_explanation)

        # Step 11: Visualize the reasoning chain.
        logging.info("Visualizing the reasoning chain...")
        visualize_reasoning_chain(best_chain, title="Abductive Reasoning Path")

    except Exception as e:
        logging.error(f"An error occurred in the main execution: {e}")

if __name__ == "__main__":
    main()
