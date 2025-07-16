from langchain.chains import ConversationalRetrievalChain
from app.chains. __init__ import llm, retriever
from app.prompts.qa_prompt import partial_prompt
from app.utils.memory_manager import get_user_memory

def get_qa_chain_for_user(user_id):
    memory = get_user_memory(user_id)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": partial_prompt},
        memory=memory
    )
