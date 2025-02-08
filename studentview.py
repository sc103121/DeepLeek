import streamlit as st

# Set up the page
st.set_page_config(page_title="Ed Discussion Clone", page_icon="🎓", layout="wide")

# Initialize session state for courses and threads if not already present
if "courses" not in st.session_state:
    # Each course is a dict mapping thread titles to a list of messages.
    st.session_state.courses = {
        "Math 101": {
            "General Discussion": [
                {"role": "assistant", "message": "Welcome to Math 101 General Discussion!"}
            ],
            "Homework Help": [
                {"role": "assistant", "message": "Welcome to Math 101 Homework Help!"}
            ],
        },
        "History 202": {
            "General Discussion": [
                {"role": "assistant", "message": "Welcome to History 202 General Discussion!"}
            ],
        },
    }

# Set defaults for the current course and thread
if "current_course" not in st.session_state:
    st.session_state.current_course = list(st.session_state.courses.keys())[0]

if "current_thread" not in st.session_state:
    default_threads = st.session_state.courses[st.session_state.current_course]
    st.session_state.current_thread = list(default_threads.keys())[0]

# Sidebar: Course selection
with st.sidebar:
    st.header("Courses")
    courses_list = list(st.session_state.courses.keys())
    selected_course = st.radio("Select a Course", courses_list, index=courses_list.index(st.session_state.current_course))
    st.session_state.current_course = selected_course

# Main layout: split into two columns.
# Left column for thread selection/creation, right column for messages.
col_threads, col_chat = st.columns([1, 3])

with col_threads:
    st.subheader("Threads")
    # Get list of threads for the current course
    threads = list(st.session_state.courses[st.session_state.current_course].keys())
    selected_thread = st.selectbox("Select Thread", threads, index=threads.index(st.session_state.current_thread))
    st.session_state.current_thread = selected_thread

    # New Thread Creation
    new_thread_title = st.text_input("New Thread Title", key="new_thread")
    if st.button("Create New Thread"):
        if new_thread_title and new_thread_title not in st.session_state.courses[st.session_state.current_course]:
            st.session_state.courses[st.session_state.current_course][new_thread_title] = [
                {"role": "assistant", "message": f"Welcome to {new_thread_title}!"}
            ]
            st.session_state.current_thread = new_thread_title
            st.experimental_rerun()
        else:
            st.error("Please enter a unique thread title.")

with col_chat:
    st.subheader(f"Thread: {st.session_state.current_thread}")
    # Retrieve messages for the selected thread
    messages = st.session_state.courses[st.session_state.current_course][st.session_state.current_thread]
    for msg in messages:
        # Use Streamlit's built-in chat_message for ChatGPT-like styling
        st.chat_message(msg["role"]).write(msg["message"])

    # Chat input area
    user_input = st.chat_input("Type your message here...")
    if user_input:
        # Append the user's message
        st.session_state.courses[st.session_state.current_course][st.session_state.current_thread].append(
            {"role": "user", "message": user_input}
        )
        # For demonstration, we simulate an assistant response by echoing the input.
        bot_response = f"You said: {user_input}"
        st.session_state.courses[st.session_state.current_course][st.session_state.current_thread].append(
            {"role": "assistant", "message": bot_response}
        )
        st.experimental_rerun()
