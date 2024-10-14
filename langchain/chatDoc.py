from langchain.document_loaders import TextLoader, DirectoryLoader, PyMuPDFLoader

def read_file(file_name):
    f = open(file_name, "r", encoding="utf-8")
    file_doc = f.read()

    if file_name.endswith(".txt"):  
        loader = TextLoader(file_doc)
        
    elif file_name.endswith(".pdf"):
        loader = PyMuPDFLoader(file_doc)

    return loader

if __name__ == "__main__":
    loader = read_file("langchain\\files\\test.txt")
    docs = loader.load() 
    print(docs)

