from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import Together
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain import LLMChain
import os
import json

with open('config_info/assistant_config.json', 'r', encoding='utf-8') as f:
    bot_config = json.load(f)

assistant_name = bot_config['assistant_name']
company_name = bot_config['company_name']
role = bot_config['role']
rules = bot_config['rules']
rules_str = "\n    - " + "\n    - ".join(rules)

embedding = HuggingFaceEmbeddings(model="BAAI/bge-large-en-v1.5" ,encode_kwargs={"normalize_embeddings": True})
db = FAISS.load_local("faiss_index_test", embedding,allow_dangerous_deserialization=True)

load_dotenv()
api_key = os.getenv("together_api_key")


llm = Together(
    model="lgai/exaone-3-5-32b-instruct",
    temperature=0,
    together_api_key=api_key
)

retriever = db.as_retriever(search_kwargs={"k": 10, "fetch_k": 10})
prompt = PromptTemplate(
    input_variables=["context", "question", "assistant_name", "company_name", "role", "rules_str"],
    template="""
You are {assistant_name}, a {role} for {company_name}. You are speaking to an employee of {company_name}.

Follow these rules strictly:
    {rules_str}

Context:
{context}

Question:
{question}

return the answer without the "answer:" text
"""
)

partial_prompt = prompt.partial(
    assistant_name=assistant_name,
    company_name=company_name,
    role=role,
    rules_str=rules_str
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
  - Just a greeting, thank you, acknowledgment,a short polite response or a bit longer response of acknowledgment→ "ack"
  - A meaningful or content-based reply that answers the question asked by the assistant in the last message where the question can be part of the statement provided by the assistant → "real"
- if there is no question asked by the assistant in the last message, then it is "ack"
---

Chat History:
{chat_history}

User Message: "{query}"

Only respond with "ack" or "real".
"""
)

ack_response_prompt = PromptTemplate(
    input_variables=["query","assistant_name","company_name","role"],
    template="""
You are {assistant_name}, a {role} at {company_name}.

The user has sent a greeting, acknowledgment, or polite message. Your job is to respond appropriately — warmly, professionally, and in a way that reflects what they said.
For Example:
- If the user says "thanks", respond with a natural variation like "You're very welcome!" or "Glad I could help!"
- If they say "hi" or "hello", greet them back and ask how you can assist.
- If they say "okay", "cool", "great", etc., acknowledge it appropriately without sounding robotic.
- Your response must feel like a human reply to what they actually said.

User said:
"{query}"

{assistant_name}'s reply:
"""
)


intent_chain = LLMChain(llm=llm, prompt=intent_prompt)

ack_chain = LLMChain(llm=llm, prompt=ack_response_prompt)


ConversationBufferWindowMemory(k=2)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    k=2
)
user_memories = {}
def get_user_memory(user_id: str):
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            k=2
        )
    return user_memories[user_id]
    
 








