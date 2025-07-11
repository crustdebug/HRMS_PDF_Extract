


from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import sentence_transformers
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from tqdm import tqdm
from langchain.llms import Together
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


import time
from functools import wraps

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function '{func.__name__}' executed in {end - start:.4f} seconds")
        return result
    return wrapper


embedding = HuggingFaceEmbeddings(model="BAAI/bge-large-en-v1.5" ,encode_kwargs={"normalize_embeddings": True})
db = FAISS.load_local("faiss_index_updated", embedding,allow_dangerous_deserialization=True)


from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain import LLMChain
import os

load_dotenv()
api_key = os.getenv("together_api_key")


llm = Together(
    model="lgai/exaone-3-5-32b-instruct",
    temperature=0,
    together_api_key=api_key
)


retriever = db.as_retriever(search_kwargs={"k": 5, "fetch_k": 10})
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an intelligent and professional HR assistant for a corporate organization named National Highways Logistics Management Ltd (NHLML) and you are named as Tara.
You are talking to the employee of NHLML about whom you don't know anything except this.

Your responsibilities include:
- Answering the question with the help of the policy.
- Do not answer questions which are not related to HR or company policies.(e.g., food, pets, travel, entertainment,or any such casual request etc.).
- If the input is just a greeting or acknowledgment (e.g., "hi", "thanks", "okay"), respond politely without referencing any documents or context.
- If you detect spelling errors in the user's question and ONLY if the user's question is in english, first ask "Did you mean [corrected spelling]?" and wait for confirmation before providing the answer.
- You are supposed to refer to the whole context first and refer to chat history ONLY if needed. DO NOT refer to the chat history if it is not needed.
- Never tell the user to refer to the policy as you are already provided with it and are supposed to answer according to it.

Context:
{context}

Question:
{question}


Answer:
"""
)


intent_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are an intent classification assistant.

Classify the user's message into one of two categories:

- "ack" → If the message is ONLY a greeting, thanks, acknowledgment, or generic affirmation (e.g. "hi", "thanks", "okay", "great").
- "real" → If the message includes a real question or request (even if it starts with a greeting).

Message: "{query}"

Respond with only "ack" or "real".
"""
)

ack_response_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are Tara, a professional but friendly HR assistant at National Highways Logistics Management Ltd (NHLML).

The user has sent a greeting, acknowledgment, or polite message. Your job is to respond appropriately — warmly, professionally, and in a way that reflects what they said.
For Example:
- If the user says "thanks", respond with a natural variation like "You're very welcome!" or "Glad I could help!"
- If they say "hi" or "hello", greet them back and ask how you can assist.
- If they say "okay", "cool", "great", etc., acknowledge it appropriately without sounding robotic.
- Your response must feel like a human reply to what they actually said.

User said:
"{query}"

Tara's reply:
"""
)


intent_chain = LLMChain(llm=llm, prompt=intent_prompt)

ack_chain = LLMChain(llm=llm, prompt=ack_response_prompt)



memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    k=5
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,  
    retriever=retriever,
    combine_docs_chain_kwargs={"prompt": prompt},
    memory=memory
 )









