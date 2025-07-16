from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pprint import pformat
from __init__ import intent_chain, ack_chain, assistant_name, company_name, role, rules, rules_str, get_user_memory
import re
from pydantic import BaseModel
from HRMSchatbot import get_user_memory, llm, retriever, partial_prompt
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from functools import partial

class Query(BaseModel):
    user_id: str  # Add this
    query: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def is_gibberish(text):
    # A basic heuristic: text is gibberish if it contains no spaces and isn't found in dictionary
    return len(text.strip()) > 4 and not re.search(r"\s", text) and not re.search(r"[aeiouAEIOU]", text)


def get_qa_chain_for_user(user_id):
    memory = get_user_memory(user_id)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,  
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": partial_prompt},
        memory=memory
    )

@app.post("/chat")
async def chat(query: Query):
    try:
        user_input = query.query.strip()
        user_id = query.user_id

        memory = get_user_memory(user_id)
        qa_chain = get_qa_chain_for_user(user_id)
        qa_chain.memory = memory

        if is_gibberish(user_input):
            return JSONResponse(content={"response": "It seems like that was a typo. Could you please rephrase your question?"})

        intent = intent_chain.run(query=user_input,chat_history=memory.buffer).strip().lower()

        if intent == "ack":
            memory.clear()
            response = ack_chain.run(query=user_input,assistant_name=assistant_name,company_name=company_name,role=role)
            
        else:
            response = qa_chain.run(question=user_input)

        if isinstance(response, (dict, list)):
            formatted_output = pformat(response)
        else:
            formatted_output = str(response)

        return JSONResponse(content={"response": formatted_output})

    except Exception as e:
        return {"error": str(e)}
