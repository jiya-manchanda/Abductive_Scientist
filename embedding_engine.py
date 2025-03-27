from sentence_transformers import SentenceTransformer, util

class EmbeddingEngine:
    def __init__(self, concept_file="data/concepts.txt"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.concepts, self.definitions = self.load_concepts(concept_file)
        self.embeddings = self.model.encode(self.definitions, convert_to_tensor=True)

    def load_concepts(self, path):
        concepts = []
        definitions = []
        with open(path, 'r') as f:
            for line in f:
                if ':' in line:
                    name, definition = line.strip().split(':', 1)
                    concepts.append(name.strip())
                    definitions.append(definition.strip())
        return concepts, definitions

    def get_similar_concepts(self, observation, top_k=5):
        obs_embedding = self.model.encode(observation, convert_to_tensor=True)
        similarities = util.cos_sim(obs_embedding, self.embeddings)[0]
        ranked = sorted(zip(self.concepts, similarities), key=lambda x: -x[1])
        return ranked[:top_k]