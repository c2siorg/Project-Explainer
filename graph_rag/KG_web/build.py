import Scraper
import pickle

base_url = input("Enter the URL of the documentation to scrape: ")

scraped_documents = Scraper.scrape_website(base_url)

for i, doc in enumerate(scraped_documents, 1):  
    print(f"\nDocument {i}")
    print(f"URL: {doc.metadata.get('url', 'No URL found')}")
    print(f"Content Snippet: {doc.text[:200]}...\n")

index = Scraper.build_index_from_documents(scraped_documents)

with open("documentation_index.pkl", "wb") as file:
    pickle.dump(index, file)

print("\nIndexing complete. The index has been saved to 'documentation_index.pkl'.")
