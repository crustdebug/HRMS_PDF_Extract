from langchain.memory import ConversationBufferMemory

user_memories = {}
def get_user_memory(user_id: str):
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            k=2
        )
    return user_memories[user_id]