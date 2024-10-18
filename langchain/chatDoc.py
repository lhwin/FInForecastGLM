from langchain.document_loaders import TextLoader, PyMuPDFLoader
from langchain.storage import LocalFileStore
from langchain.chains import StuffDocumentsChain, LLMChain
# from langchain.document_loaders import GithubFileLoader
from langchain.text_splitter import TextSplitter, CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import os

api_key = "sk-uLPxcqRQ4r9eFvDD7225F90eEa0f479dB2D918E6AbBfF934"
api_base = "https://api.openai99.top/v1"

os.environ["OPENAI_API_BASE"] = api_base
os.environ["OPENAI_API_KEY"] = api_key

e_model = OpenAIEmbeddings()
fs = LocalFileStore("./cache/")
local_cache = CacheBackedEmbeddings.from_bytes_store(e_model, fs, namespace=e_model.model)

llm = OpenAI(model="gpt-3.5-turbo-16k",
            temperature=0)

document_prompt = PromptTemplate(input_variable=["page_content"], template="{page_content}")

stuff_prompt_override = """Given this text extracts:
    __________________________________________

    {context}

    ------------------------------------------

    Please answer the following questions:
    {query}

    """
prompt = PromptTemplate(template=stuff_prompt_override,
                            input_variable=["context", "query"])

llmchain = LLMChain(llm=llm, prompt=prompt)

workchain = StuffDocumentsChain(llm_chain=llmchain, document_prompt=document_prompt,
                                    document_variable_name="context")


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

    db.save_local("./cache/faiss_index")

    return db

def query2gpt(query, db_path="./cache/faiss_index"):
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

    doc = db.similarity_search_with_score(query)

    document = [d[0] for d in doc]
    res = workchain.run({'input_documents':document, "query": query})

    print(res)
    return res
def main():
    file_path = "./files/test.txt"

    # docs = read_and_split(file_path)

    query2gpt("依萍怎么了?")

    # print(docs)


if __name__ == "__main__":
    main()

