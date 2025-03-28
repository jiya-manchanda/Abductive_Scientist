# embedding_engine.py

from sentence_transformers import SentenceTransformer, util
import logging

# Configure logging for debugging and informational output.
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class EmbeddingEngine:
    """
    An embedding engine that leverages SentenceTransformer to generate semantic embeddings for concepts and facts.
    
    This module loads concept definitions and factual statements from text files, computes their embeddings,
    and provides methods to retrieve the most semantically related concepts and facts given an observation.
    """
    def __init__(self, concept_file="data/concepts.txt", fact_file="data/facts.txt"):
        """
        Initialize the EmbeddingEngine.

        Loads the pre-trained SentenceTransformer model and reads in concepts and facts from files.
        Computes embeddings for the concept definitions and facts.

        Parameters:
            concept_file (str): Path to the file containing concept definitions.
            fact_file (str): Path to the file containing factual statements.
        """
        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            logging.info("SentenceTransformer model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading SentenceTransformer model: {e}")
            raise e

        # Load concepts and their definitions.
        self.concepts, self.definitions = self.load_concepts(concept_file)
        try:
            self.concept_embeddings = self.model.encode(self.definitions, convert_to_tensor=True)
            logging.info("Concept embeddings computed successfully.")
        except Exception as e:
            logging.error(f"Error encoding concept definitions: {e}")
            self.concept_embeddings = None

        # Load facts and compute their embeddings.
        self.facts = self.load_facts(fact_file)
        try:
            self.fact_embeddings = self.model.encode(self.facts, convert_to_tensor=True)
            logging.info("Fact embeddings computed successfully.")
        except Exception as e:
            logging.error(f"Error encoding facts: {e}")
            self.fact_embeddings = None

    def load_concepts(self, path):
        """
        Load concept names and definitions from a specified file.
        
        The expected file format is one concept per line, with the concept name and definition
        separated by a colon (":"). For example:
            memory_loss: A decline in the ability to encode, store, or retrieve information.

        Parameters:
            path (str): Path to the concepts file.

        Returns:
            tuple: Two lists - one containing concept names and the other containing definitions.
        """
        concepts = []
        definitions = []
        try:
            with open(path, 'r') as f:
                for line in f:
                    if ':' in line:
                        try:
                            name, definition = line.strip().split(':', 1)
                            concepts.append(name.strip())
                            definitions.append(definition.strip())
                        except Exception as e:
                            logging.error(f"Error parsing concept line '{line}': {e}")
        except Exception as e:
            logging.error(f"Failed to load concepts from {path}: {e}")
        logging.info("Loaded %d concepts from %s", len(concepts), path)
        return concepts, definitions

    def load_facts(self, path):
        """
        Load factual statements from a specified file.

        Each non-empty line in the file should contain a fact.
        
        Parameters:
            path (str): Path to the facts file.
        
        Returns:
            list: A list of fact strings.
        """
        try:
            with open(path, 'r') as f:
                facts = [line.strip() for line in f if line.strip()]
            logging.info("Loaded %d facts from %s", len(facts), path)
            return facts
        except Exception as e:
            logging.error(f"Failed to load facts from {path}: {e}")
            return []

    def get_related_concepts(self, observation, top_k=5):
        """
        Retrieve the top-k concepts that are semantically related to the given observation.

        Parameters:
            observation (str): The text observation to compare against.
            top_k (int): The number of top related concepts to return.

        Returns:
            list: A list of concept names, ranked by semantic similarity.
        """
        try:
            obs_embedding = self.model.encode(observation, convert_to_tensor=True)
            similarities = util.cos_sim(obs_embedding, self.concept_embeddings)[0]
            ranked = sorted(zip(self.concepts, similarities), key=lambda x: -x[1])
            return [concept for concept, score in ranked[:top_k]]
        except Exception as e:
            logging.error(f"Error computing related concepts: {e}")
            return []

    def get_related_facts(self, observation, top_k=3):
        """
        Retrieve the top-k factual statements that are semantically related to the given observation.

        Parameters:
            observation (str): The text observation to compare against.
            top_k (int): The number of top related facts to return.

        Returns:
            list: A list of fact strings, ranked by semantic similarity.
        """
        try:
            obs_embedding = self.model.encode(observation, convert_to_tensor=True)
            similarities = util.cos_sim(obs_embedding, self.fact_embeddings)[0]
            ranked = sorted(zip(self.facts, similarities), key=lambda x: -x[1])
            return [fact for fact, score in ranked[:top_k]]
        except Exception as e:
            logging.error(f"Error computing related facts: {e}")
            return []

if __name__ == "__main__":
    # For debugging: test the EmbeddingEngine with a sample observation.
    engine = EmbeddingEngine()
    observation = "memory loss and impaired learning"
    logging.info("Testing related concepts and facts for observation: '%s'", observation)
    related_concepts = engine.get_related_concepts(observation)
    related_facts = engine.get_related_facts(observation)
    
    print("Related Concepts:")
    for concept in related_concepts:
        print(f" - {concept}")
    
    print("\nRelated Facts:")
    for fact in related_facts:
        print(f" - {fact}")
