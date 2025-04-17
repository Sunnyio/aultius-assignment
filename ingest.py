import os
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from urllib.parse import urljoin

DATA_DIR = "data"

def fetch_web_pages(base_url="https://www.angelone.in/support"):
    from urllib.parse import urljoin
    visited = set()
    all_texts = []

    print(f"Fetching base page: {base_url}")
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, "html.parser")
    base_links = soup.find_all("a", href=True)

    base_text = soup.get_text(separator="\n", strip=True)
    all_texts.append(base_text)
    visited.add(base_url)

    for a in base_links:
        href = a['href']
        full_url = urljoin(base_url, href)
        if full_url.startswith(base_url) and full_url not in visited:
            print(f"Fetching: {full_url}")
            try:
                sub_resp = requests.get(full_url)
                sub_soup = BeautifulSoup(sub_resp.text, "html.parser")
                sub_text = sub_soup.get_text(separator="\n", strip=True)
                all_texts.append(sub_text)
                visited.add(full_url)
            except:
                pass

    # Save all to files
    os.makedirs(f"{DATA_DIR}/web_pages", exist_ok=True)
    for i, content in enumerate(all_texts):
        with open(f"{DATA_DIR}/web_pages/page_{i}.txt", "w", encoding="utf-8") as f:
            f.write(content)
    return 'visited: ', visited

def load_documents():
    docs = []

    # Load web text files
    for filename in os.listdir(f"{DATA_DIR}/web_pages"):
        path = os.path.join(DATA_DIR, "web_pages", filename)
        loader = TextLoader(path)
        docs.extend(loader.load())

    # Load PDFs
    for filename in os.listdir(f"{DATA_DIR}/pdfs"):
        if filename.endswith(".pdf"):
            path = os.path.join(DATA_DIR, "pdfs", filename)
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
            print(filename)
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

if __name__ == "__main__":
    fetch_web_pages()
    docs = load_documents()
    chunks = split_documents(docs)
    print(f"Total chunks: {len(chunks)}")
