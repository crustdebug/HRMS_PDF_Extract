from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import Together
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain import LLMChain
from app.prompts.qa_prompt import qa_prompt, partial_prompt
from app.prompts.intent_prompt import intent_prompt
from app.prompts.ack_response_prompt import ack_response_prompt
from app.chains.config import assistant_name, company_name, role, rules_str
from app.utils.memory_manager import get_user_memory

import os

embedding = HuggingFaceEmbeddings(model="thenlper/gte-large" ,encode_kwargs={"normalize_embeddings": True})
db = FAISS.load_local("data/faiss_index_gemini_CO_gtelarge", embedding,allow_dangerous_deserialization=True)

load_dotenv()
api_key = os.getenv("together_api_key")

llm = Together(
    model="lgai/exaone-3-5-32b-instruct",
    temperature=0.1
)  

retriever = db.as_retriever(search_kwargs={"k": 10, "fetch_k": 15})

intent_chain = LLMChain(llm=llm, prompt=intent_prompt)
ack_chain = LLMChain(llm=llm, prompt=ack_response_prompt)

ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True,k=2)