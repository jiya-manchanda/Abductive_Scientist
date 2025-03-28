# embedding_engine.py

from sentence_transformers import SentenceTransformer, util

class EmbeddingEngine:
    def __init__(self, concept_file="data/concepts.txt", fact_file="data/facts.txt"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.concepts, self.definitions = self.load_concepts(concept_file)
        self.concept_embeddings = self.model.encode(self.definitions, convert_to_tensor=True)
        self.facts = self.load_facts(fact_file)
        self.fact_embeddings = self.model.encode(self.facts, convert_to_tensor=True)

    def load_concepts(self, path):
        """Load concept names and definitions."""
        concepts = []
        definitions = []
        with open(path, 'r') as f:
            for line in f:
                if ':' in line:
                    name, definition = line.strip().split(':', 1)
                    concepts.append(name.strip())
                    definitions.append(definition.strip())
        return concepts, definitions

    def load_facts(self, path):
        """Load list of factual knowledge statements."""
        with open(path, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    def get_related_concepts(self, observation, top_k=5):
        """Return top-k concepts most semantically related to the observation."""
        obs_embedding = self.model.encode(observation, convert_to_tensor=True)
        similarities = util.cos_sim(obs_embedding, self.concept_embeddings)[0]
        ranked = sorted(zip(self.concepts, similarities), key=lambda x: -x[1])
        return [c for c, _ in ranked[:top_k]]

    def get_related_facts(self, observation, top_k=3):
        """Return top-k facts most semantically related to the observation."""
        obs_embedding = self.model.encode(observation, convert_to_tensor=True)
        similarities = util.cos_sim(obs_embedding, self.fact_embeddings)[0]
        ranked = sorted(zip(self.facts, similarities), key=lambda x: -x[1])
        return [f for f, _ in ranked[:top_k]]

    # Optionally, you can add methods to update the embeddings if the dataset changes.
    
# For debugging:
if __name__ == "__main__":
    engine = EmbeddingEngine()
    observation = "memory loss and impaired learning"
    print("Related concepts:")
    print(engine.get_related_concepts(observation))
    print("Related facts:")
    print(engine.get_related_facts(observation))
