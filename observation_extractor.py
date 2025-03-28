# observation_extractor.py

from transformers import pipeline

# Initialize the summarization pipeline (this may download the model on first run)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_observations(text):
    """
    Extract key observation(s) from a given abstract or text.
    Returns a summarized observation.
    """
    # Adjust parameters as needed based on your typical abstract length.
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    return summary[0]['summary_text']

# For debugging:
if __name__ == "__main__":
    sample_text = ("Recent studies indicate that prolonged stress may lead to hippocampal atrophy, "
                   "resulting in impaired memory encoding and retrieval. Experimental data further "
                   "suggest that reduced levels of BDNF are a contributing factor.")
    obs = extract_observations(sample_text)
    print("Extracted Observation:")
    print(obs)
