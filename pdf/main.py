import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import os

load_dotenv()

def get_store_path(store_name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stores', f'{store_name}.pkl')

def create_vector_store(text):
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    # Word embedding
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def extract_text_from_pdf(pdf):
    pdf_reader = PdfReader(pdf)

    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

CHAR_LIMIT = 1000

def main():
    st.header('Chat with PDF')
    st.write('Please do not upload sensitive information. This app is for demonstration purposes only.')

    pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input('Enter a query')

    if not pdf:
        st.info('Please upload a PDF file')
        return

    if pdf:
        if query:
            st.info(query)

        # Read the PDF file
        text = extract_text_from_pdf(pdf)

        if len(text.split()) >= CHAR_LIMIT:
            st.error(f'Text is too long, please upload a shorter PDF (less than {CHAR_LIMIT} words)')
            return

        vector_store = create_vector_store(text)
        
        if query:
            docs = vector_store.similarity_search(query, 5)
            llm = OpenAI(temperature=0, max_tokens=250)
            chain = load_qa_chain(llm, chain_type='stuff')

            response = chain.run(input_documents=docs, question=query)

            st.write(response)


if __name__ == "__main__":
    main()
