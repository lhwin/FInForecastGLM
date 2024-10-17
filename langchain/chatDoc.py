from langchain.document_loaders import TextLoader, PyMuPDFLoader
from langchain.storage import LocalFileStore
# from langchain.document_loaders import GithubFileLoader
from langchain.text_splitter import TextSplitter, CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
api_key = "sk-uLPxcqRQ4r9eFvDD7225F90eEa0f479dB2D918E6AbBfF934"
api_base = "https://api.openai99.top/v1"

e_model = OpenAIEmbeddings(api_key=api_key, base_url=api_base)
fs = LocalFileStore("./cache/")
local_cache = CacheBackedEmbeddings.from_bytes_store(e_model, fs, namespace=e_model.model)

def read_and_split(file_name):

    if file_name.endswith(".txt"):
        loader = TextLoader(file_name, encoding="utf-8")
        raw_documents = loader.load()
    elif file_name.endswith(".pdf"):
        loader = PyMuPDFLoader(file_name)
        raw_documents = loader.load()

    text_spliter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    text = text_spliter.split_documents(raw_documents)
    db = FAISS.from_documents(text, local_cache)

    return text

def main():
    file_path = "./files/test.txt"

    docs = read_and_split(file_path)
    print(docs)


if __name__ == "__main__":
    main()

