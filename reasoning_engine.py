# Symbolic Module: SymbolicReasoner
# Performs logical reasoning and abductive inference over causal rules

import networkx as nx
import matplotlib.pyplot as plt
from sentence_transformers import util

class SymbolicReasoner:
    def __init__(self, rules_path):
        self.rules = self.load_rules(rules_path)
        self.graph = self.build_graph()

    def load_rules(self, path):
        rules = []
        with open(path, 'r') as f:
            for line in f:
                if '=>' in line:
                    premise, conclusion = map(str.strip, line.strip().split('=>'))
                    rules.append((premise, conclusion))
        return rules

    def build_graph(self):
        G = nx.DiGraph()
        for premise, conclusion in self.rules:
            G.add_edge(premise, conclusion)
        return G

    def explain(self, target, depth=3):
        paths = []
        self._trace_explanation(target, [], paths, depth)
        return paths

    def _trace_explanation(self, target, current_path, all_paths, depth):
        if depth == 0:
            return
        for premise, conclusion in self.rules:
            if conclusion == target:
                new_path = [(premise, conclusion)] + current_path
                all_paths.append(new_path)
                self._trace_explanation(premise, new_path, all_paths, depth - 1)

    def connect_concepts(self, concepts):
        chains = []
        for i in range(len(concepts)):
            for j in range(len(concepts)):
                if i != j:
                    try:
                        path = nx.shortest_path(self.graph, source=concepts[i], target=concepts[j])
                        edge_chain = [(path[k], path[k + 1]) for k in range(len(path) - 1)]
                        chains.append(edge_chain)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
        return chains

    def score_chain(self, chain, known_facts, user_input, embedder):
        # Import locally to prevent circular import
        from reasoning_engine import explain_chain_naturally

        explanation_text = explain_chain_naturally(chain)
        input_embed = embedder.model.encode(user_input, convert_to_tensor=True)
        chain_embed = embedder.model.encode(explanation_text, convert_to_tensor=True)
        sim_score = float(util.cos_sim(input_embed, chain_embed)[0])

        fact_match = sum(1 for p, c in chain for f in known_facts if p in f or c in f)

        # Weighted scoring: adjust weights as needed
        return 0.5 * len(chain) + 1.0 * fact_match + 2.0 * sim_score

    def select_best_explanation(self, concept_list, known_facts, user_input, embedder):
        best_chain = None
        best_score = -1
        all_chains = []

        for concept in concept_list:
            chains = self.explain(concept)
            for chain in chains:
                score = self.score_chain(chain, known_facts, user_input, embedder)
                all_chains.append((chain, score))
                if score > best_score:
                    best_score = score
                    best_chain = chain

        return best_chain, best_score, all_chains


def explain_chain_naturally(chain):
    if not chain:
        return "No explanation found."

    steps = [
        f"Because {premise.replace('_', ' ')}, it may lead to {conclusion.replace('_', ' ')}."
        for premise, conclusion in chain
    ]
    summary = f"Therefore, the observed issue may ultimately be due to {chain[0][0].replace('_', ' ')}."
    return "\n".join(steps + [summary])


def visualize_reasoning_chain(chain, title="Reasoning Path"):
    if not chain:
        print("No reasoning chain to visualize.")
        return

    G = nx.DiGraph()
    for premise, conclusion in chain:
        G.add_edge(premise.replace('_', ' '), conclusion.replace('_', ' '))

    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 6))
    nx.draw(
        G, pos, with_labels=True, node_size=2000, node_color="lightblue",
        edge_color="gray", font_size=10, font_weight='bold'
    )
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    