import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# -----------------------------------
# Page settings
# -----------------------------------
st.set_page_config(
    page_title="AI College FAQ Assistant",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 AI College FAQ Assistant")
st.write("Ask your college-related questions here.")

# -----------------------------------
# Load FAQ dataset
# -----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("college_faq.csv")

df = load_data()

# Check required columns
required_columns = {"question", "answer", "category"}
if not required_columns.issubset(df.columns):
    st.error("CSV file must contain these columns: question, answer, category")
    st.stop()

# -----------------------------------
# Load AI model
# -----------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------------
# Create question embeddings
# -----------------------------------
@st.cache_resource
def create_embeddings(question_list):
    return model.encode(question_list, convert_to_tensor=True)

questions = df["question"].astype(str).tolist()
question_embeddings = create_embeddings(questions)

# -----------------------------------
# Function to get answer
# -----------------------------------
def get_best_answer(user_question, threshold=0.78):
    user_question = user_question.strip()

    if not user_question:
        return "Please enter a question."

    # Convert user question to embedding
    user_embedding = model.encode(user_question, convert_to_tensor=True)

    # Compare with all dataset questions
    scores = util.cos_sim(user_embedding, question_embeddings)[0]

    # Get best match
    best_match_idx = torch.argmax(scores).item()
    best_score = scores[best_match_idx].item()

    # If similarity is too low, show custom message
    if best_score < threshold:
        return "Sorry, I couldn’t find an answer for that question. Please ask a college-related question."

    # Otherwise return the matched answer
    return df.iloc[best_match_idx]["answer"]

# -----------------------------------
# Chat history
# -----------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------------
# Input box
# -----------------------------------
user_question = st.text_input("Enter your question:")

if st.button("Ask"):
    if user_question.strip():
        answer = get_best_answer(user_question)
        st.session_state.chat_history.append(("You", user_question))
        st.session_state.chat_history.append(("Bot", answer))
    else:
        st.warning("Please enter a question.")

# -----------------------------------
# Show chat messages
# -----------------------------------
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(
            f"""
            <div style="background-color:#dbeafe; padding:12px; border-radius:12px; margin:8px 0;">
                <b>🧑 You:</b> {message}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="background-color:#f3f4f6; padding:12px; border-radius:12px; margin:8px 0;">
                <b>🤖 Bot:</b> {message}
            </div>
            """,
            unsafe_allow_html=True
        )

