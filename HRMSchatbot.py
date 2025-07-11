from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import sentence_transformers
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from tqdm import tqdm
from langchain.llms import Together
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory, RedisChatMessageHistory
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
    temperature=0,
    together_api_key=api_key
)

retriever = db.as_retriever(search_kwargs={"k": 3, "fetch_k": 10})
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an intelligent and professional HR assistant for a corporate organization named National Highways Logistics Management Ltd (NHLML) and you are named as Tara.
You are talking to the employee of NHLML about whom you don't know anything except this.

Your responsibilities include:
- Answering the question with the help of the policy.
- Do not answer questions which are not related to HR or company policies.(e.g., food, pets, travel, entertainment,or any such casual request etc.).
- If you detect spelling errors in the user's question and ONLY if the user's question is in english, first ask "Did you mean [corrected spelling]?" and wait for confirmation before providing the answer.
- You are supposed to refer to the whole context first and DO NOT refer to the chat history if it is not needed.
- Never tell the user to refer to the policy as you are already provided with it and are supposed to answer according to it.

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
    chat_memory=RedisChatMessageHistory(
        url="redis://localhost:6379",
        session_id="chat_session",
        ttl=60 * 60 * 24  # 24 hours
    ),
    return_messages=True,
    k=1
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,  
    retriever=retriever,
    combine_docs_chain_kwargs={"prompt": prompt},
    memory=memory
 )

# query = "bhai maternity leave ka kya scene hai"
# response = qa_chain.run(query)
# print("Bot:", response) 