from langchain.document_loaders import PDFPlumberLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.llms import Ollama
import streamlit as st
from langchain_text_splitters import TextSplitter

ollama = Ollama(base_url='http://localhost:11434', model="llama3")
def load_file(file):
    if file.endswith(".txt") or file.endswith(".md"):
        loader = TextLoader(file)
        doc = loader.load()
    elif file.endswith("pdf"):
        loader = PDFPlumberLoader(file)
        doc = loader.load_and_split()

    return doc


def file_split(file, mode="text"):
    f = open(file, "r", encoding="utf-8")
    ff = f.read()

    if mode == "text":
        text_spliter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20, length_function=len, add_start_index=True)
        doc = text_spliter([ff])

    else:
        text_spliter = CharacterTextSplitter(chunk_size=50, chunk_overlap=20, length_function=len, add_start_index=True, is_separator_regex=False)
        doc=text_spliter.create_documents([ff])

    return doc

