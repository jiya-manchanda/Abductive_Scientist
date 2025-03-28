# dataset_manager.py

import logging

# Configure logging for debugging and informational output.
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def update_facts(new_fact, fact_file="data/facts.txt"):
    """
    Update the facts dataset by appending a new fact if it is not already present.
    
    This function reads the current facts from the specified file, checks if the new_fact
    already exists, and if not, appends it to the file. It logs the action taken.
    
    Parameters:
        new_fact (str): The new fact to be added to the dataset.
        fact_file (str): The path to the file where facts are stored. Defaults to "data/facts.txt".
    
    Returns:
        None
    """
    try:
        # Attempt to read existing facts from the file.
        with open(fact_file, "r") as f:
            facts = f.read().splitlines()
    except FileNotFoundError:
        logging.warning("Facts file not found. A new file will be created.")
        facts = []
    except Exception as e:
        logging.error(f"Error reading facts file: {e}")
        return

    # Check if the new fact is already in the dataset.
    if new_fact in facts:
        logging.info("Fact already exists in the dataset.")
    else:
        try:
            with open(fact_file, "a") as f:
                # Append the new fact to the file, ensuring it's on a new line.
                f.write(f"\n{new_fact}")
            logging.info("New fact added to the dataset.")
        except Exception as e:
            logging.error(f"Error writing new fact to file: {e}")

# For debugging: Test the update_facts function.
if __name__ == "__main__":
    test_fact = "New observation: stress increases cortisol levels."
    update_facts(test_fact)
