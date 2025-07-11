from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import sentence_transformers
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from tqdm import tqdm
from langchain.llms import Together
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import os
from langchain.document_loaders import TextLoader

embedding = HuggingFaceEmbeddings(model="BAAI/bge-large-en-v1.5" ,encode_kwargs={"normalize_embeddings": True})
db = FAISS.load_local("faiss_index_updated", embedding,allow_dangerous_deserialization=True)



load_dotenv()
api_key = os.getenv("together_api_key")

llm = Together(
    model="lgai/exaone-3-5-32b-instruct",
    temperature=0.5,
    together_api_key=api_key
)

retriever = db.as_retriever(search_kwargs={"k": 3, "fetch_k": 10})
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an intelligent and professional HR assistant for a corporate organization named National Highways Logistics Management Ltd (NHLML) and you are named is Tara.
You are talking to the employee of NHLML about whom you don't know anything except this.

Your responsibilities include:
- Answering the question according to the policy
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

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt}
)

# query = "bhai maternity leave ka kya scene hai"
# response = qa_chain.run(query)
# print("Bot:", response) 