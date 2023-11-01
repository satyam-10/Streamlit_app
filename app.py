import streamlit as st
import faiss
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader

from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings


def process_llm_response(llm_response):
    response = llm_response['result']
    source_documents = llm_response['source_documents']
    return response, source_documents


st.title("ChatX OC v2 Chatbot")


user_query = st.text_input("Enter your question:")


uploaded_file = st.file_uploader("Upload a document (optional)", type=["pdf", "txt"])


text_splitter = RecursiveCharacterTextSplitter(
                                               chunk_size=1000,
                                               chunk_overlap=200)

texts = text_splitter.split_documents(uploaded_file)

selected_model = st.radio("Select a Model", ["Instructor Embeddings", "OpenAI Embeddings"])
from langchain.embeddings import HuggingFaceInstructEmbeddings
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                      model_kwargs={"device": "cuda"})

db_instructEmbedd = FAISS.from_documents(texts, instructor_embeddings)
retriever = db_instructEmbedd.as_retriever(search_kwargs={"k": 3})
qa_chain_instrucEmbed = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.2, ),
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)

if st.button("Submit"):
    st.write(f"User Query: {user_query}")

    if selected_model == "Instructor Embeddings":
        llm_response = qa_chain_instrucEmbed(user_query)  # You should define qa_chain_instrucEmbed
    else:
        llm_response = qa_chain_openai(user_query)  # You should define qa_chain_openai

    response, source_documents = process_llm_response(llm_response)

    st.write("Response:")
    st.write(response)

    st.write("Sources:")
    for source in source_documents:
        st.write(source.metadata['source'])

