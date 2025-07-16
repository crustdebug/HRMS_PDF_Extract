from langchain.prompts import PromptTemplate
from app.chains.config import assistant_name, company_name, role, rules_str

qa_prompt = PromptTemplate(
    input_variables=["context", "question", "assistant_name", "company_name", "role", "rules_str"],
    template="""
You are {assistant_name}, a {role} for {company_name}. You are speaking to an employee of {company_name}.

Follow these rules strictly:
    {rules_str}

Context:
{context}

Question:
{question}

Return the answer without the "Answer:" prefix.
"""
)

# Apply partials so the chain only needs to pass context and question
partial_prompt = qa_prompt.partial(
    assistant_name=assistant_name,
    company_name=company_name,
    role=role,
    rules_str=rules_str
)
