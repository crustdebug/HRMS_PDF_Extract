from langchain.prompts import PromptTemplate
from app.chains.config import assistant_name, company_name, role

ack_response_prompt = PromptTemplate(
    input_variables=["query", "assistant_name", "company_name", "role"],
    template="""
You are {assistant_name}, a {role} at {company_name}.

The user has sent a greeting, acknowledgment, or polite message. Your job is to respond appropriately — warmly, professionally, and in a way that reflects what they said.

Examples:
- If the user says "thanks", respond with: "You're very welcome!" or "Glad I could help!"
- If they say "hi" or "hello", greet them and ask how you can assist.
- If they say "okay", "cool", "great", etc., acknowledge it in a natural way.

User said:
"{query}"

Tara's reply:
"""
)
