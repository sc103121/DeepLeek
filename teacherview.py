import streamlit as st

# Set up the page with a title, icon, and layout
st.set_page_config(page_title="ChatGPT Clone with History", page_icon="💬", layout="wide")

# Initialize session state for multiple chat sessions if not already present.
if "chat_sessions" not in st.session_state:
    # Start with one chat session called "Chat 1" with a welcome message.
    st.session_state.chat_sessions = {
        "Chat 1": [{"role": "assistant", "message": "Hello, I'm ChatGPT. How can I help you today?"}]
    }
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Chat 1"

# Sidebar: Chat history selection and new chat creation
with st.sidebar:
    st.header("Chat Histories")
    # List of existing chat sessions
    chat_list = list(st.session_state.chat_sessions.keys())
    # Radio button to select a chat session
    selected_chat = st.radio("Select Chat", chat_list, index=chat_list.index(st.session_state.current_chat))
    st.session_state.current_chat = selected_chat

    # Button to create a new chat session
    if st.button("New Chat"):
        new_chat_name = f"Chat {len(chat_list) + 1}"
        st.session_state.chat_sessions[new_chat_name] = [
            {"role": "assistant", "message": "Hello, I'm ChatGPT. How can I help you today?"}
        ]
        st.session_state.current_chat = new_chat_name
        st.experimental_rerun()  # Rerun to update the sidebar with the new chat

# Main chat container for the selected chat session
current_chat = st.session_state.chat_sessions[st.session_state.current_chat]

# Display each message using the built-in chat_message component
for msg in current_chat:
    st.chat_message(msg["role"]).write(msg["message"])

# Chat input: use the built-in chat_input component
user_input = st.chat_input("Type your message here...")

if user_input:
    # Append the user's message to the current chat session
    st.session_state.chat_sessions[st.session_state.current_chat].append({"role": "user", "message": user_input})
    
    # Here you could integrate your AI logic to generate a response.
    # For demonstration, we simply echo the user's input.
    bot_response = f"You said: {user_input}"
    st.session_state.chat_sessions[st.session_state.current_chat].append({"role": "assistant", "message": bot_response})
    
    # Rerun to update the UI with the new messages
    st.experimental_rerun()
