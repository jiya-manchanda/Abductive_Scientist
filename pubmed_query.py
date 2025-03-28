# pubmed_query.py

import requests
from xml.etree import ElementTree as ET

def fetch_pubmed_data(query, max_results=5):
    """
    Query the PubMed API for the given query string, retrieve up to max_results
    articles, and parse the XML to extract titles, abstracts, authors, publication
    dates, and PubMed links.
    """
    # 1. eSearch to get WebEnv and QueryKey
    search_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
        f"db=pubmed&term={query}&retmax={max_results}&usehistory=y"
    )
    search_response = requests.get(search_url)
    if search_response.status_code != 200:
        print(f"Error fetching data from PubMed (search step): {search_response.status_code}")
        return []

    search_tree = ET.fromstring(search_response.text)
    webenv_elem = search_tree.find(".//WebEnv")
    query_key_elem = search_tree.find(".//QueryKey")

    # If either is missing, we can't proceed
    if webenv_elem is None or query_key_elem is None:
        print("No WebEnv or QueryKey found in search response.")
        return []

    webenv = webenv_elem.text
    query_key = query_key_elem.text

    # 2. eFetch to retrieve detailed articles
    fetch_url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
        f"db=pubmed&query_key={query_key}&WebEnv={webenv}&retmode=xml"
    )
    fetch_response = requests.get(fetch_url)
    if fetch_response.status_code != 200:
        print(f"Error fetching data from PubMed (fetch step): {fetch_response.status_code}")
        return []

    fetch_tree = ET.fromstring(fetch_response.text)
    papers = []

    # PubMed articles are in <PubmedArticle> elements
    for article in fetch_tree.findall(".//PubmedArticle"):
        # PMID
        pmid_elem = article.find(".//PMID")
        pmid = pmid_elem.text if pmid_elem is not None else ""

        # Title
        title_elem = article.find(".//ArticleTitle")
        title_text = title_elem.text if title_elem is not None else "No title available"

        # Abstract
        abstract_elem = article.find(".//AbstractText")
        abstract_text = abstract_elem.text if abstract_elem is not None else "No abstract available"

        # Authors
        authors = []
        for author in article.findall(".//Author"):
            last_name_elem = author.find("LastName")
            fore_name_elem = author.find("ForeName")
            if last_name_elem is not None and fore_name_elem is not None:
                authors.append(f"{fore_name_elem.text} {last_name_elem.text}")
            elif last_name_elem is not None:
                authors.append(last_name_elem.text)

        # Publication date (very approximate, you may want to refine this)
        pub_date_elem = article.find(".//PubDate")
        pub_date_text = pub_date_elem.text if pub_date_elem is not None else "Unknown"

        # Construct a PubMed link
        link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}" if pmid else ""

        papers.append({
            "pmid": pmid,
            "title": title_text,
            "abstract": abstract_text,
            "authors": authors,
            "published": pub_date_text,
            "link": link
        })

    return papers

# Example usage:
if __name__ == "__main__":
    # Test with a known query (e.g., "memory hippocampus")
    query = "memory hippocampus"
    results = fetch_pubmed_data(query, max_results=3)
    for i, paper in enumerate(results, 1):
        print(f"\nPaper #{i}")
        print(f"PMID: {paper['pmid']}")
        print(f"Title: {paper['title']}")
        print(f"Abstract: {paper['abstract']}")
        print(f"Authors: {', '.join(paper['authors']) if paper['authors'] else 'N/A'}")
        print(f"Published: {paper['published']}")
        print(f"Link: {paper['link']}")
