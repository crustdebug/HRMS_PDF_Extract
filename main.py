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
    return len(text.strip()) > 4 and not re.search(r"\s", text) and not re.search(r"[aeiouAEIOU]", text)

ACKNOWLEDGEMENTS = [
    "ok", "okay", "thanks", "thank you", "got it", "understood", "noted",
    "yes", "yep", "yeah", "alright", "roger", "sure", "fine", "cool", "great", "perfect", "sounds good"
]

# Acknowledgement detector
def is_genuine_acknowledgement(message):
    # Normalize message
    msg = message.strip().lower()
    # Remove punctuation for better matching
    msg = re.sub(r'[^\w\s]', '', msg)
    # Check for exact match or phrase match
    for ack in ACKNOWLEDGEMENTS:
        if msg == ack or msg.startswith(ack + " ") or msg.endswith(" " + ack) or ack in msg.split():
            return True
    return False

@app.post("/chat")
async def chat(query: Query):
    try:
        response = qa_chain.run(query.query.strip().lower())

        if isinstance(response, (dict, list)):
            formatted_output = pformat(response)
        else:
            formatted_output = str(response)

        return JSONResponse(content={"response": formatted_output})

    except Exception as e:
        return {"error": str(e)}
