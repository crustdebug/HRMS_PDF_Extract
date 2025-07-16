from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pprint import pformat
from app.models.query_model import Query
from app.chains import intent_chain, ack_chain, assistant_name, company_name, role
from app.chains.qa_chain import get_qa_chain_for_user
from app.utils.gibberish import is_gibberish
from app.utils.memory_manager import get_user_memory

router = APIRouter()

@router.post("/chat")
async def chat(query: Query):
    try:
        user_input = query.query.strip()
        user_id = query.user_id

        memory = get_user_memory(user_id)
        qa_chain = get_qa_chain_for_user(user_id)
        qa_chain.memory = memory

        if is_gibberish(user_input):
            return JSONResponse(content={"response": "It seems like that was a typo. Could you please rephrase your question?"})

        intent = intent_chain.run(query=user_input, chat_history=memory.buffer).strip().lower()

        if intent == "ack":
            memory.clear()
            response = ack_chain.run(
                query=user_input,
                assistant_name=assistant_name,
                company_name=company_name,
                role=role
            )
        else:
            response = qa_chain.run(question=user_input)

        formatted_output = pformat(response) if isinstance(response, (dict, list)) else str(response)
        return JSONResponse(content={"response": formatted_output})
        
    except Exception as e:
        return {"error": str(e)}