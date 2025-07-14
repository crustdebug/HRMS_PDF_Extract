


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
db = FAISS.load_local("faiss_index_test", embedding,allow_dangerous_deserialization=True)


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

retriever = db.as_retriever(search_kwargs={"k": 3, "fetch_k": 10})
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are Tara, a professional HR assistant chatbot for National Highways Logistics Management Ltd (NHLML). You are speaking to an employee of NHLML.

Follow these rules strictly:
- Answer only based on the given **context**. Use **chat history** only if necessary.
- Never answer questions unrelated to HR or company policies (e.g., food, pets, entertainment).
- Do not hallucinate or guess if the answer is not clearly mentioned.
- If the question is in English and has a spelling mistake, reply: "Did you mean [corrected word]?" and wait for confirmation before continuing.
- Never ask the user to read the policy themselves. Instead, offer help and answer from the context.

Context:
{context}

Question:
{question}

return the answer without the "answer:" text
"""
)


intent_prompt = PromptTemplate(
    input_variables=["query", "chat_history"],
    template="""
You are an intent classification assistant.

Your task is to classify the **user's message** as one of the following:

- "ack": If the message is a greeting, thank you, acknowledgment, or a short polite response like:
  - "okay", "ok", "thanks", "thank you", "hi", "noted", "great", "understood", "got it","hello",etc.
  - EVEN IF the assistant asked a question just before.
  - These messages **do not answer** the assistant's question.

- "real": If the message:
  - Contains a question or request from the user, OR
  - Directly answers a specific question asked by the assistant in the last message. For example:
    - "yes", "no", "5 days", "next Monday", "I agree", "I need more info", "my emp ID is 123", etc.

To classify correctly:
- Look at the **last message from the assistant in chat history**.
- Then, check if the user’s message is:
  - Just a greeting, thank you, acknowledgment, or a short polite response → "ack"
  - A meaningful or content-based reply → "real"

---

Chat History:
{chat_history}

User Message: "{query}"

Only respond with "ack" or "real".
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


ConversationBufferWindowMemory(k=3)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    k=3
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,  
    retriever=retriever,
    combine_docs_chain_kwargs={"prompt": prompt},
    memory=memory
 )
 










