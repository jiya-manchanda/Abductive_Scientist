import requests
from xml.etree import ElementTree as ET

def fetch_pubmed_data(query, max_results=5):
    """
    Query the PubMed API and fetch metadata (title, abstract, authors, etc.) for a given search term.
    """
    # Step 1: Search PubMed for the query
    search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query}&retmax={max_results}&usehistory=y"
    
    response = requests.get(search_url)
    if response.status_code != 200:
        print(f"Error fetching data from PubMed: {response.status_code}")
        return []
    
    # Parse the search results (Get WebEnv and QueryKey)
    tree = ET.fromstring(response.text)
    webenv = tree.find(".//WebEnv").text
    query_key = tree.find(".//QueryKey").text

    # Step 2: Fetch the detailed data using efetch
    fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&query_key={query_key}&WebEnv={webenv}&retmode=xml"
    
    fetch_response = requests.get(fetch_url)
    if fetch_response.status_code != 200:
        print(f"Error fetching detailed articles: {fetch_response.status_code}")
        return []
    
    # Parse the fetched data
    fetch_tree = ET.fromstring(fetch_response.text)
    papers = []
    for docsum in fetch_tree.findall(".//DocSum"):
        title = docsum.find(".//Item[@Name='Title']").text
        authors = [author.text for author in docsum.findall(".//Item[@Name='Author']")]
        pub_date = docsum.find(".//Item[@Name='PubDate']").text
        source = docsum.find(".//Item[@Name='Source']").text
        link = f"https://pubmed.ncbi.nlm.nih.gov/{source.split('/')[-1]}" if source else ""
        
        # Add the article details to the papers list
        papers.append({
            "title": title,
            "summary": "No summary provided",  # PubMed doesn't provide summary directly, you can scrape the abstract if needed
            "authors": authors,
            "published": pub_date,
            "link": link
        })

    return papers

# Example Usage:
query = "memory loss"  # This can be dynamic based on user input
papers = fetch_pubmed_data(query, max_results=3)

for paper in papers:
    print(f"Title: {paper['title']}")
    print(f"Summary: {paper['summary']}")
    print(f"Authors: {', '.join(paper['authors'])}")
    print(f"Published: {paper['published']}")
    print(f"Link: {paper['link']}")
    print("="*50)
