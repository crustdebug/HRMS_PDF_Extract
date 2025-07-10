from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pprint import pformat
from HRMSchatbot import qa_chain  
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
    return len(text.strip()) > 6 and not re.search(r"\s", text) and not re.search(r"[aeiouAEIOU]", text)

ACK_KEYWORDS = [
    "ok", "okay", "thanks", "thank you", "cool", "alright", "sure", 
    "noted", "got it", "fine", "great", "awesome", "nice", "understood", 
    "done", "perfect", "k", "yup", "yes"
]

# Acknowledgement detector
def is_acknowledgement(text: str) -> bool:
    cleaned = text.strip().lower()
    cleaned = re.sub(r'[^\w\s]', '', cleaned)
    for word in ACK_KEYWORDS:
        if word in cleaned:
            return True
    return False


@app.post("/chat")
async def chat(query: Query):
    try:
        user_input = query.query.strip().lower()

        if is_gibberish(user_input):
            return JSONResponse(content={"response": "It seems like that was a typo. Could you please rephrase your question?"})
        
        if is_acknowledgement(user_input):
            qa_chain.memory.clear()
            return JSONResponse(content={"response": "Do you have any other request?"})

        response = qa_chain.run(query.query.strip().lower())

        if isinstance(response, (dict, list)):
            formatted_output = pformat(response)
        else:
            formatted_output = str(response)

        return JSONResponse(content={"response": formatted_output})

    except Exception as e:
        return {"error": str(e)}
