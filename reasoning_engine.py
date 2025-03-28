import os
# Suppress Tk deprecation warnings.
os.environ["TK_SILENCE_DEPRECATION"] = "1"

import warnings
# Suppress the tight_layout warning about incompatible Axes.
warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")

import matplotlib
# Optionally switch the backend to reduce macOS-specific logs (choose one that works best in your environment).
matplotlib.use("TkAgg")

import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import util
import logging

# Configure logging to output debug and informational messages with timestamps.
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class SymbolicReasoner:
    """
    A symbolic reasoning engine that uses a rule-based approach to generate explanations.
    It loads rules from a file, constructs a directed graph, and then performs reasoning
    by tracing paths through the graph.
    """
    def __init__(self, rules_path):
        """
        Initialize the SymbolicReasoner.
        
        Parameters:
            rules_path (str): Path to a text file containing rules. Each line should have
                              the format: premise => conclusion
        """
        self.rules = self.load_rules(rules_path)
        self.graph = self.build_graph()

    def load_rules(self, path):
        """
        Load the rules from a given file.
        
        Reads each line from the file, parses it to extract the premise and conclusion,
        and stores the rules as a list of tuples.
        
        Parameters:
            path (str): Path to the file containing rules.
        
        Returns:
            list of tuple: A list where each element is a (premise, conclusion) tuple.
        """
        rules = []
        try:
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Ensure the line is non-empty and follows the expected format.
                    if line and '=>' in line:
                        try:
                            premise, conclusion = map(str.strip, line.split('=>'))
                            rules.append((premise, conclusion))
                        except Exception as e:
                            logging.error(f"Error parsing line '{line}': {e}")
        except Exception as e:
            logging.error(f"Failed to load rules from {path}: {e}")
        return rules

    def build_graph(self):
        """
        Build a directed graph from the loaded rules.
        
        Each rule is added as an edge in the graph from the premise to the conclusion.
        
        Returns:
            networkx.DiGraph: A directed graph representing the rules.
        """
        G = nx.DiGraph()
        for premise, conclusion in self.rules:
            G.add_edge(premise, conclusion)
        logging.info("Graph built with %d nodes and %d edges", G.number_of_nodes(), G.number_of_edges())
        return G

    def explain(self, target, depth=3):
        """
        Generate all possible explanations for a target concept by recursively tracing rules.
        
        Parameters:
            target (str): The concept to generate explanations for.
            depth (int): The maximum depth to trace back the reasoning.
        
        Returns:
            list of list: A list of reasoning chains. Each chain is a list of (premise, conclusion) tuples.
        """
        paths = []
        self._trace_explanation(target, [], paths, depth)
        return paths

    def _trace_explanation(self, target, current_path, all_paths, depth):
        """
        Recursively trace back through the rules to generate reasoning chains.
        
        This private method appends complete reasoning chains to the all_paths list when
        a rule leading to the target is found. It stops when the maximum recursion depth is reached.
        
        Parameters:
            target (str): The current target concept.
            current_path (list): The current chain of reasoning (list of tuples).
            all_paths (list): The master list that accumulates all reasoning chains.
            depth (int): Remaining recursion depth.
        """
        if depth == 0:
            return  # Base case: maximum depth reached, stop recursion.
        found = False
        # Iterate over each rule to see if it concludes to the current target.
        for premise, conclusion in self.rules:
            if conclusion == target:
                # Build a new chain by adding this rule.
                new_path = [(premise, conclusion)] + current_path
                all_paths.append(new_path)
                # Recursively trace back using the premise as the new target.
                self._trace_explanation(premise, new_path, all_paths, depth - 1)
                found = True
        if not found:
            # If no rule leads to the target, this branch terminates.
            return

    def connect_concepts(self, concepts):
        """
        Find and return the shortest paths (chains) between every pair of concepts.
        
        Parameters:
            concepts (list): A list of concept names (strings).
        
        Returns:
            list of list: A list of chains, where each chain is a list of (node1, node2) edge tuples.
        """
        chains = []
        # Compare every pair of concepts.
        for i in range(len(concepts)):
            for j in range(len(concepts)):
                if i != j:
                    try:
                        # Find the shortest path from one concept to another.
                        path = nx.shortest_path(self.graph, source=concepts[i], target=concepts[j])
                        # Convert the path into a list of edge tuples.
                        edge_chain = [(path[k], path[k + 1]) for k in range(len(path) - 1)]
                        chains.append(edge_chain)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue  # Skip pairs where no path exists.
        return chains

    def score_chain(self, chain, known_facts, user_input, embedder):
        """
        Compute a score for a reasoning chain based on its length, matching known facts,
        and semantic similarity with the user input.
        
        Parameters:
            chain (list): A list of (premise, conclusion) tuples representing the chain.
            known_facts (list): A list of known facts (strings) to match against the chain.
            user_input (str): The user's input used for semantic similarity scoring.
            embedder: An object with a 'model' attribute for generating embeddings.
        
        Returns:
            float: A score representing the quality of the reasoning chain.
        """
        # Generate a natural language explanation from the chain.
        from reasoning_engine import explain_chain_naturally  # Local import to avoid circular dependencies.
        explanation_text = explain_chain_naturally(chain)
        try:
            # Encode the user input and the explanation text to tensors.
            input_embed = embedder.model.encode(user_input, convert_to_tensor=True)
            chain_embed = embedder.model.encode(explanation_text, convert_to_tensor=True)
            # Compute cosine similarity between the two embeddings.
            sim_score = float(util.cos_sim(input_embed, chain_embed)[0])
        except Exception as e:
            logging.error(f"Error computing embeddings: {e}")
            sim_score = 0.0

        # Count how many parts of the chain match any known fact.
        fact_match = sum(1 for p, c in chain for f in known_facts if p in f or c in f)
        # Compute final score: shorter chains and better fact matches and similarity yield a higher score.
        score = 0.5 * len(chain) + 1.0 * fact_match + 2.0 * sim_score
        return score

    def select_best_explanation(self, concept_list, known_facts, user_input, embedder):
        """
        Evaluate all possible reasoning chains generated from a list of concepts and select the best one.
        
        Parameters:
            concept_list (list): List of concept names (strings) to generate explanations for.
            known_facts (list): List of known facts for additional scoring.
            user_input (str): The original user input for semantic similarity scoring.
            embedder: An object with a 'model' attribute for generating embeddings.
        
        Returns:
            tuple: (best_chain, best_score, all_chains)
                   best_chain: The reasoning chain with the highest score.
                   best_score: The score of the best chain.
                   all_chains: A list of all chains paired with their respective scores.
        """
        best_chain = None
        best_score = -1
        all_chains = []

        # Iterate over each concept and generate explanation chains.
        for concept in concept_list:
            chains = self.explain(concept)
            for chain in chains:
                score = self.score_chain(chain, known_facts, user_input, embedder)
                all_chains.append((chain, score))
                if score > best_score:
                    best_score = score
                    best_chain = chain

        logging.info("Selected best chain with score: %.2f", best_score)
        return best_chain, best_score, all_chains

def explain_chain_naturally(chain):
    """
    Convert a reasoning chain into a human-readable explanation.
    
    Parameters:
        chain (list): A list of (premise, conclusion) tuples.
    
    Returns:
        str: A natural language explanation of the reasoning chain.
    """
    if not chain:
        return "No explanation found."
    # Generate a series of step-by-step explanations.
    steps = [
        f"Because {premise.replace('_', ' ')}, it may lead to {conclusion.replace('_', ' ')}."
        for premise, conclusion in chain
    ]
    # Provide a concluding summary.
    summary = f"Therefore, the observed issue may ultimately be due to {chain[0][0].replace('_', ' ')}."
    return "\n".join(steps + [summary])

def visualize_reasoning_chain(chain, title="Reasoning Path"):
    """
    Visualize a reasoning chain using a directed graph representation.
    
    Parameters:
        chain (list): A list of (premise, conclusion) tuples.
        title (str): Title for the plot.
    
    Displays:
        A matplotlib plot of the reasoning chain as a graph.
    """
    if not chain:
        logging.warning("No reasoning chain to visualize.")
        return

    # Create a directed graph for the reasoning chain.
    G = nx.DiGraph()
    for premise, conclusion in chain:
        # Replace underscores with spaces for better readability.
        G.add_edge(premise.replace('_', ' '), conclusion.replace('_', ' '))
    pos = nx.spring_layout(G)  # Position nodes using spring layout for clarity.
    plt.figure(figsize=(10, 6))
    # Draw nodes and edges with labels.
    nx.draw(
        G, pos, with_labels=True, node_size=2000, node_color="lightblue",
        edge_color="gray", font_size=10, font_weight='bold'
    )
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20)
    plt.title(title)
    try:
        plt.tight_layout()
    except Exception as e:
        logging.warning("tight_layout encountered an issue: %s", e)
    plt.show()

if __name__ == "__main__":
    # Debugging mode: run the reasoner on a sample target.
    logging.info("Starting SymbolicReasoner for debugging purposes.")
    reasoner = SymbolicReasoner("data/scientific_rules.txt")
    sample_explanations = reasoner.explain("memory_loss")
    for chain in sample_explanations:
        print(explain_chain_naturally(chain))
