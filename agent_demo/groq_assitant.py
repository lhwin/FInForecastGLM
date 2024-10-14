from typing import List, Optional

import streamlit as st
from phi.assistant import Assistant
from phi.knowledge import AssistantKnowledge
from phi.document import Document
from phi.document.reader.pdf import PDFReader
from phi.document.reader.website import WebsiteReader
from phi.utils.log import logger

from phi.llm.groq import Groq
from phi.embedder.openai import OpenAIEmbedder
from phi.embedder.ollama import OllamaEmbedder
from phi.vectordb.pgvector import PgVector2
from phi.storage.assistant.postgres import PgAssistantStorage

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
st.set_page_config(
    page_title="Groq RAG",
    page_icon=":orange_heart:",
)

st.title("Agent Rag")
st.markdown("##### :orange_heart")

def get_rag_assistant(
    llm_model: str = "gpt-4-turbo",
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    debug_mode: bool = True,
    num_documents: Optional[int] = None,
) -> Assistant:
    """Get a RAG Assistant with SingleStore backend."""

    logger.info(f"-*- Creating RAG Assistant. LLM: {llm_model} -*-")

    if llm_model.startswith("gpt"):
        return Assistant(
            name="singlestore_rag_assistant",
            run_id=run_id,
            user_id=user_id,
            llm=OpenAIChat(model=llm_model),
            knowledge_base=AssistantKnowledge(
                vector_db=S2VectorDb(
                    collection="rag_documents_openai",
                    schema=DATABASE,
                    db_engine=db_engine,
                    embedder=OpenAIEmbedder(model="text-embedding-3-small", dimensions=1536),
                ),
                num_documents=num_documents,
            ),
            description="You are an AI called 'SQrL' designed to assist users in the best way possible",
            instructions=[
                "When a user asks a question, first search your knowledge base using `search_knowledge_base` tool to find relevant information.",
                "Carefully read relevant information and provide a clear and concise answer to the user.",
                "You must answer only from the information in the knowledge base.",
                "Share links where possible and use bullet points to make information easier to read.",
                "Do not use phrases like 'based on my knowledge' or 'depending on the information'.",
                "Keep your conversation light hearted and fun.",
                "Always aim to please the user",
            ],
            show_tool_calls=True,
            search_knowledge=True,
            read_chat_history=True,
            # This setting adds chat history to the messages list
            add_chat_history_to_messages=True,
            # Add 6 messages from the chat history to the messages list
            num_history_messages=6,
            add_datetime_to_instructions=True,
            # -*- Disable storage in the start
            # storage=S2AssistantStorage(table_name="auto_rag_assistant_openai", schema=DATABASE, db_engine=db_engine),
            markdown=True,
            debug_mode=debug_mode,
        )
    else:
        llm: LLM = Ollama(model=llm_model)
        if llm_model == "llama3-70b-8192":
            llm = Groq(model=llm_model)

        return Assistant(
            name="singlestore_rag_assistant",
            run_id=run_id,
            user_id=user_id,
            llm=llm,
            knowledge_base=AssistantKnowledge(
                vector_db=S2VectorDb(
                    collection="rag_documents_nomic",
                    schema=DATABASE,
                    db_engine=db_engine,
                    embedder=OllamaEmbedder(model="nomic-embed-text", dimensions=768),
                ),
                num_documents=num_documents,
            ),
            description="You are an AI called 'SQrL' designed to assist users in the best way possible",
            instructions=[
                "When a user asks a question, you will be provided with relevant information to answer the question.",
                "Carefully read relevant information and provide a clear and concise answer to the user.",
                "You must answer only from the information in the knowledge base.",
                "Share links where possible and use bullet points to make information easier to read.",
                "Keep your conversation light hearted and fun.",
                "Always aim to please the user",
            ],
            # This setting will add the references from the vector store to the prompt
            add_references_to_prompt=True,
            add_datetime_to_instructions=True,
            markdown=True,
            debug_mode=debug_mode,
            # -*- Disable memory to save on tokens
            # This setting adds chat history to the messages
            # add_chat_history_to_messages=True,
            # num_history_messages=4,
            # -*- Disable storage in the start
            # storage=S2AssistantStorage(table_name="auto_rag_assistant_ollama", schema=DATABASE, db_engine=db_engine),
        )

def get_grop_assistant(
        llm_model: str = "llama3-70b-8192",
        embedding_model: str = "text-embedding-3-small",
        user_id: Optional[str] = None,
        run_id: Optional[str] = None,
        debug_mode: bool = True
    ):
    embedder = (
        OllamaEmbedder(model=embedding_model, dimensions=768)
        if embedding_model == "nomic-embed-text"
        else OpenAIEmbedder(model=embedding_model)
    )

    embedding_table = ("groq_rag_documents_ollama" if embedding_model == "nomic-embed-text" else "groq_rag_documents_openai")

    return Assistant(
        name="groq_rag_assistant",
        run_id=run_id,
        user_id=user_id,
        llm=Groq(model=llm_model),
        storage=PgAssistantStorage(table_name="groq_rag_assistant", db_url=db_url),

        knowledge_base=AssistantKnowledge(
            vector_db=PgVector2(
                db_url=db_url,
                collection=embedding_table,
                embedder=embedder
            )
        ),
        num_documents=2,
    )
def restart_assistant():
    st.session_state["rag_assistant"] = None
    st.session_state["rag_assistant"] = None

    if "url_scrape_key" in st.session_state:
        st.session_state["url_scrape_key"] += 1

    if "file_uploder_key" in st.session_state:
        st.session_state["file_uploder_key"] += 1

    st.rerun()

def main():
    llm_model = st.sidebar.selectbox("Select LLM", options=["llama3-70b-8192", "llama3-8b-8192"])

    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = llm_model

    elif st.session_state["llm_model"] != llm_model:
        st.session_state["llm_model"] = llm_model
        restart_assistant()

    embedding_model = st.sidebar.selectbox(
        "select Embeddings",
        options=["nomic-embed-text", "text-embedding-3-small"]
    )

    if "embedding_model" not in st.session_state:
        st.session_state["embedding_model"] = embedding_model

    elif st.session_state["embedding_model"] != llm_model:
        st.session_state["embedding_model"] = embedding_model
        st.session_state["embedding_model_update"] = True
        restart_assistant()

        rag_assistant: Assistant
        if "rag_assistant" not in st.session_state or st.session_state["rag_assistant"] is None:
            rag_assistant = get_grop_assistant(llm_model, embedding_model)
        else:
            rag_assistant = st.session_state["rag_assistant"]

        try:
            st.session_state["rag_assistant"]=rag_assistant.create_run()
        except Exception:
            st.warning("could not run rag assistant is the databaseruning?")

        assitant_chat_history = rag_assistant.memory.get_chat_history()
        if len(assitant_chat_history) > 0:
            logger.debug("loading chat history")
            st.session_state["messages"] = assitant_chat_history
        else:
            logger.debug("no chat history found")
            st.session_state["messgaes"] = [{"role": "assistant", "content":"please upload a file and ask me question"}]

            if prompt :=st.chat_input():
                st.session_state["messages"].append({"role":"user", "content":prompt})

            for message in st.session_state["messages"]:
                if message["role"] == "system":
                    continue
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            last_message = st.session_state["messages"][-1]
            if last_message.get("role") == "user":
                question = last_message["content"]
                with st.chat_message("assistant"):
                    response = ""
                    resp_container = st.empty()
                    for delta in rag_assistant.run(question):
                        response += delta
                        resp_container.markdown(response)
                    st.session_state["message"].append({"role":"assistant", "content":response})

                #载入知识库
                if rag_assistant.knowledge_base:
                    if "url_scrapy_key" not in st.session_state:
                        st.session_state["url_scrapy_key"] = 0

                    input_url = st.sidebar.text_input(
                        "Add URL to Knowledge Base", type="default", key=st.session_state["url_scrape_key"]
                    )

                    add_url_button = st.sidebar.button("Add url")
                    if add_url_button:
                        if input_url is not None:
                            alert = st.sidebar.info("Processing URLs...")

                            if f"{input_url}_scarped" not in st.session_state:
                                scraper = WebsiteReader(max_links=2, max_depth=1)
                                web_documents:List[Document] = scraper.read(input_url)

                                if web_documents:
                                    rag_assistant.knowledge_base.load(web_documents, upsert=True)
                                else:
                                    st.sidebar.error("couldn't find web")
                                st.session_state[f"{input_url}_upload"]
                            alert.empty

                    if "file_uploader_key" not in st.session_state:
                        st.session_state["file_uploader_key"] = 100

                    uploaded_file = st.sidebar.file_uploader(
                        "Add a PDF :page_facing_up:", type="pdf", key=st.session_state["file_uploader_key"]
                    )
                    if uploaded_file is not None:
                        alert = st.sidebar.info("Processing PDF...", icon="🧠")
                        rag_name = uploaded_file.name.split(".")[0]
                        if f"{rag_name}_uploaded" not in st.session_state:
                            reader = PDFReader()
                            rag_documents: List[Document] = reader.read(uploaded_file)
                            if rag_documents:
                                rag_assistant.knowledge_base.load_documents(rag_documents, upsert=True)
                            else:
                                st.sidebar.error("Could not read PDF")
                            st.session_state[f"{rag_name}_uploaded"] = True
                        alert.empty()
            if rag_assistant.knowledge_base and rag_assistant.knowledge_base.vector_db:
                if st.sidebar.button("Clear Knowledge Base"):
                    rag_assistant.knowledge_base.vector_db.clear()
                    st.sidebar.success("Knowledge base cleared")

            if rag_assistant.storage:
                rag_assistant_run_ids: List[str] = rag_assistant.storage.get_all_run_ids()
                new_rag_assistant_run_id = st.sidebar.selectbox("Run ID", options=rag_assistant_run_ids)
                if st.session_state["rag_assistant_run_id"] != new_rag_assistant_run_id:
                    logger.info(f"---*--- Loading {llm_model} run: {new_rag_assistant_run_id} ---*---")
                    st.session_state["rag_assistant"] = get_rag_assistant(
                        llm_model=llm_model, embeddings_model=embedding_model, run_id=new_rag_assistant_run_id
                    )
                    st.rerun()

            if st.sidebar.button("New Run"):
                restart_assistant()

if __name__ == "__main__":
    main()