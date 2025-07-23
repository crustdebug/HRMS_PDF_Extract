from langchain.prompts import PromptTemplate

intent_prompt = PromptTemplate(
    input_variables=["query", "chat_history"],
    template="""
You are an intent classification assistant.

Your task is to classify the **user's message** as one of the following:

- "ack": If the message is a greeting, thank you, acknowledgment, or a short polite response like:
  - "okay", "ok", "thanks", "thank you", "hi", "noted", "great", "understood", "got it", "hello", etc.
  - EVEN IF the assistant asked a question just before.
  - These messages **do not answer** the assistant's question.

- "real": If the message:
  - Contains a question or request from the user, OR
  - Directly answers a specific question asked by the assistant in the last message. For example:
    - "yes", "no", "5 days", "next Monday", "I agree", "I need more info", "my emp ID is 123", etc.

To classify correctly:
- Look at the **last message from the assistant in chat history**.
- Then, check if the user’s message is:
  - Just a greeting, thank you, acknowledgment, or a polite response → "ack"
  - A meaningful or content-based reply that answers the question asked by the assistant → "real"

Chat History:
{chat_history}

User Message: "{query}"

Only respond with "ack" or "real".
"""
)
