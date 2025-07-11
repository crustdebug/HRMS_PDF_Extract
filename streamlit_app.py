import streamlit as st
from HRMSchatbot import qa_chain, intent_chain, ack_chain

st.set_page_config(page_title="HRMS Chatbot", page_icon="🤖", layout="centered")
st.title("HR Assistant - Tara")
st.write("Ask any HR or company policy related question.")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

user_input = st.text_input("Your question:", key="input")

if st.button("Ask") or (user_input and st.session_state.get('last_input') != user_input):
    if user_input:
        st.session_state['last_input'] = user_input
        with st.spinner('Thinking...'):
            try:
                intent = intent_chain.run(query=user_input).strip().lower()
                if intent == "ack":
                    response = ack_chain.run(user_input)
                else:
                    response = qa_chain.run(user_input)
                st.session_state['chat_history'].append((user_input, response))
            except Exception as e:
                st.session_state['chat_history'].append((user_input, f"Error: {e}"))

if st.session_state['chat_history']:
    st.write("## Chat History")
    for q, a in reversed(st.session_state['chat_history']):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Tara:** {a}") 