# dataset_manager.py

def update_facts(new_fact, fact_file="data/facts.txt"):
    """
    Append a new fact to the facts file if it is not already present.
    """
    with open(fact_file, "r") as f:
        facts = f.read().splitlines()

    if new_fact not in facts:
        with open(fact_file, "a") as f:
            f.write(f"\n{new_fact}")
        print("New fact added to dataset.")
    else:
        print("Fact already exists in dataset.")

# For debugging:
if __name__ == "__main__":
    test_fact = "New observation: stress increases cortisol levels."
    update_facts(test_fact)
