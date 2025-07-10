"""import subprocess

subprocess.run(["pip", "install", "-r", "requirements.txt"])
# In[2]:
"""

import warnings
warnings.filterwarnings("ignore")
import os
import sentence_transformers
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Together
from langchain.prompts import PromptTemplate



embedding = HuggingFaceEmbeddings(model="BAAI/bge-large-en-v1.5" ,encode_kwargs={"normalize_embeddings": True})
db = FAISS.load_local("faiss_index", embedding , allow_dangerous_deserialization=True)




load_dotenv()
api_key = os.getenv("together_api_key")


llm = Together( 
    model="lgai/exaone-3-5-32b-instruct",
    together_api_key=api_key
    
)



retriever = db.as_retriever(search_kwargs={"k": 3, "fetch_k": 10})
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an intelligent and professional HR assistant for a corporate organization named National Highways Logistics Management Ltd (NHLML) and you are named is Tara.
You are talking to the employee of NHLML about whom you don't know anything except this.

Your responsibilities include:
- Understand the question and then provide accurate answers strictly
- Do not answer questions which are not related to HR or company policies.(e.g., food, pets, travel, entertainment,or any such casual request etc.).
- In case of unsurety,just reply with "Sorry that is not mentioned in the policy".
- Do not respond after simple acknowledgments like "okay", "thanks", "ok", etc.


Context:
{context}

Question:
{question}

Answer:
"""
)

ConversationBufferWindowMemory(k=2)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    k=2
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,  
    retriever=retriever,
    combine_docs_chain_kwargs={"prompt": prompt},
    memory=memory
)
