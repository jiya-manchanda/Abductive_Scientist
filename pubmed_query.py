# pubmed_query.py

import requests
from xml.etree import ElementTree as ET
import logging

# Configure logging to output messages with timestamps and log levels.
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def fetch_pubmed_data(query, max_results=5):
    """
    Query the PubMed API for a given search term and retrieve up to max_results articles.
    
    The function performs two steps:
    1. eSearch: Retrieves WebEnv and QueryKey parameters required for fetching results,
       limited to max_results.
    2. eFetch: Uses the above parameters to retrieve detailed article data, also limited
       to max_results articles.
    
    It parses the returned XML to extract key details such as PMID, title, abstract,
    authors, publication date, and constructs a PubMed link.
    
    Parameters:
        query (str): The search query string (should be URL-encoded if necessary).
        max_results (int): Maximum number of articles to fetch.
    
    Returns:
        list of dict: Each dictionary contains:
            - pmid (str): PubMed ID.
            - title (str): Title of the article.
            - abstract (str): Abstract text.
            - authors (list): List of authors.
            - published (str): Publication date (approximate).
            - link (str): URL linking to the PubMed article.
    
    Example:
        >>> papers = fetch_pubmed_data("memory hippocampus", max_results=3)
    """
    try:
        # Step 1: eSearch to get WebEnv and QueryKey parameters, limiting results to max_results.
        search_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
            f"db=pubmed&term={query}&retmax={max_results}&usehistory=y"
        )
        logging.info("Sending PubMed eSearch request...")
        search_response = requests.get(search_url)
        if search_response.status_code != 200:
            logging.error(f"Error fetching data from PubMed (search step): {search_response.status_code}")
            return []

        # Parse the XML response from the eSearch step.
        search_tree = ET.fromstring(search_response.text)
        webenv_elem = search_tree.find(".//WebEnv")
        query_key_elem = search_tree.find(".//QueryKey")
        
        # Ensure that the necessary parameters were found.
        if webenv_elem is None or query_key_elem is None:
            logging.error("No WebEnv or QueryKey found in the search response.")
            return []

        webenv = webenv_elem.text
        query_key = query_key_elem.text

        # Step 2: eFetch to retrieve detailed article information,
        # limiting the number of articles returned by including retmax in the URL.
        fetch_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
            f"db=pubmed&query_key={query_key}&WebEnv={webenv}&retmode=xml&retmax={max_results}"
        )
        logging.info("Sending PubMed eFetch request...")
        fetch_response = requests.get(fetch_url)
        if fetch_response.status_code != 200:
            logging.error(f"Error fetching data from PubMed (fetch step): {fetch_response.status_code}")
            return []

        # Parse the XML response from the eFetch step.
        fetch_tree = ET.fromstring(fetch_response.text)
        papers = []

        # Loop over each article in the response.
        for article in fetch_tree.findall(".//PubmedArticle"):
            # Extract PubMed ID (PMID)
            pmid_elem = article.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else ""

            # Extract the article title.
            title_elem = article.find(".//ArticleTitle")
            title_text = title_elem.text if title_elem is not None else "No title available"

            # Extract the abstract text.
            abstract_elem = article.find(".//AbstractText")
            abstract_text = abstract_elem.text if abstract_elem is not None else "No abstract available"

            # Extract author information.
            authors = []
            for author in article.findall(".//Author"):
                last_name_elem = author.find("LastName")
                fore_name_elem = author.find("ForeName")
                if last_name_elem is not None and fore_name_elem is not None:
                    authors.append(f"{fore_name_elem.text} {last_name_elem.text}")
                elif last_name_elem is not None:
                    authors.append(last_name_elem.text)

            # Extract the publication date (this is approximate and may need refinement).
            pub_date_elem = article.find(".//PubDate")
            pub_date_text = pub_date_elem.text if pub_date_elem is not None else "Unknown"

            # Construct a PubMed link using the PMID.
            link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}" if pmid else ""

            # Append the extracted details to the list of papers.
            papers.append({
                "pmid": pmid,
                "title": title_text,
                "abstract": abstract_text,
                "authors": authors,
                "published": pub_date_text,
                "link": link
            })

        logging.info("Fetched %d papers from PubMed.", len(papers))
        return papers

    except Exception as e:
        logging.error(f"An exception occurred while fetching PubMed data: {e}")
        return []

# For debugging: Test the fetch_pubmed_data function.
if __name__ == "__main__":
    # Sample query for testing purposes.
    query = "memory hippocampus"
    logging.info("Testing PubMed query with: '%s'", query)
    results = fetch_pubmed_data(query, max_results=5)
    for i, paper in enumerate(results, 1):
        print(f"\nPaper #{i}")
        print(f"PMID: {paper['pmid']}")
        print(f"Title: {paper['title']}")
        print(f"Abstract: {paper['abstract']}")
        print(f"Authors: {', '.join(paper['authors']) if paper['authors'] else 'N/A'}")
        print(f"Published: {paper['published']}")
        print(f"Link: {paper['link']}")
