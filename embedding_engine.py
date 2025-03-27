from sentence_transformers import SentenceTransformer, util

class EmbeddingEngine:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def get_similar_concepts(self, observation, concept_list):
        obs_embedding = self.model.encode(observation, convert_to_tensor=True)
        concept_embeddings = self.model.encode(concept_list, convert_to_tensor=True)
        similarities = util.cos_sim(obs_embedding, concept_embeddings)[0]
        ranked = sorted(zip(concept_list, similarities), key=lambda x: -x[1])
        return ranked[:3]