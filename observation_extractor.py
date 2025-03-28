# observation_extractor.py

from transformers import pipeline
import logging

# Configure logging for debugging and informational output.
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

try:
    # Initialize the summarization pipeline.
    # This model is used to generate a concise summary (observation) from a given text.
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    logging.info("Summarization pipeline initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize the summarization pipeline: {e}")
    summarizer = None

def extract_observations(text):
    """
    Extract key observations from a given text using a summarization model.
    
    This function processes the provided text (e.g., an abstract) and returns a summarized version
    that highlights the main observation or conclusion. It leverages the 'facebook/bart-large-cnn'
    summarization model.

    Parameters:
        text (str): The input text to summarize.

    Returns:
        str: A summarized observation extracted from the text. If summarization fails, returns an error message.
    
    Raises:
        ValueError: If the summarization pipeline is not initialized.
    """
    if not summarizer:
        error_msg = "Summarization pipeline is not initialized."
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        # Adjust max_length and min_length parameters as needed based on expected abstract size.
        summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
        observation = summary[0]['summary_text']
        logging.info("Observation extracted successfully.")
        return observation
    except Exception as e:
        logging.error(f"Error during summarization: {e}")
        return "Error extracting observation."

# For debugging: Run a sample observation extraction.
if __name__ == "__main__":
    sample_text = (
        "Recent studies indicate that prolonged stress may lead to hippocampal atrophy, "
        "resulting in impaired memory encoding and retrieval. Experimental data further "
        "suggest that reduced levels of BDNF are a contributing factor."
    )
    obs = extract_observations(sample_text)
    print("Extracted Observation:")
    print(obs)
