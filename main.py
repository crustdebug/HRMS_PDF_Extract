from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pprint import pformat
from HRMSchatbot import qa_chain, intent_chain, ack_chain, memory
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

def is_gibberish(text):
    # A basic heuristic: text is gibberish if it contains no spaces and isn't found in dictionary
    return len(text.strip()) > 4 and not re.search(r"\s", text) and not re.search(r"[aeiouAEIOU]", text)


@app.post("/chat")
async def chat(query: Query):
    try:
        user_input = query.query.strip()

        if is_gibberish(user_input):
            return JSONResponse(content={"response": "It seems like that was a typo. Could you please rephrase your question?"})

        intent = intent_chain.run(query=user_input,chat_history=memory.buffer).strip().lower()

        if intent == "ack":
            response = ack_chain.run(user_input)
            
        else:
            response = qa_chain.run(user_input)

        if isinstance(response, (dict, list)):
            formatted_output = pformat(response)
        else:
            formatted_output = str(response)

        return JSONResponse(content={"response": formatted_output})

    except Exception as e:
        return {"error": str(e)}
