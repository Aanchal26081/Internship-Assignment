import streamlit as st
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


#sidebar contents
with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''
                ## About
                This app is an LLM-powered chatbot built using:
                - [Streamlit]
                - [Langchain]
                - [OpenAI]
                ''')
    add_vertical_space(5)
    
    st.write("Made with Streamlit by Aanchal Gaurh")
    
def main():
    st.header("Chat with PDF")
    load_dotenv()  
    
    # Upload a pdf file
    pdf = st.file_uploader("Upload your PDF",type='pdf')
    
    if pdf is not None:
        st.write(pdf.name)
        pdf_reader = PdfReader(pdf)
        # st.write(pdf_reader)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        # st.write(text)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        # st.write(chunks)
        
        store_name = pdf.name[:-4]
        
        if os.path.exists(f"{store_name}.pk1"):
            with open(f"{store_name}.pk1","rb") as f:
                VectorStore = pickle.load(f)
            st.write('Embedding loaded from the disk')
            
        else:  
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks,embedding=embeddings)  
            with open(f"{store_name}.pk1","wb") as f:
                pickle.dump(VectorStore,f)
            st.write('Computation Completed')
        
        # Accept use Query
        query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)
        
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
 
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
    
if __name__ == '__main__':
    main()
