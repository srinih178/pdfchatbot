import streamlit as st
import requests

# Backend API URLs
UPLOAD_API_URL = "http://127.0.0.1:5000/upload_pdf"
ASK_API_URL = "http://127.0.0.1:5000/ask"

# Streamlit App
st.title("PDF Q&A Chatbot")

# File Upload Section
st.header("Upload a PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    # Upload the file to the backend
    st.info("Uploading file to the backend...")
    files = {"file": uploaded_file}
    response = requests.post(UPLOAD_API_URL, files=files)

    if response.status_code == 200:
        st.success("PDF uploaded and processed successfully!")
    else:
        st.error("Failed to upload PDF. Please try again.")

# Ask a Question Section
st.header("Ask a Question")
question = st.text_input("Enter your question")

if st.button("Get Answer"):
    if question:
        # Send the question to the backend
        st.info("Fetching the answer...")
        payload = {"question": question}
        response = requests.post(ASK_API_URL, json=payload)
        print("response", response.status_code, response.text)
        if response.status_code == 200:
            answer = response.json().get("answer", "No answer available.")
            st.success(f"Answer: {answer}")
        else:
            st.error("Failed to fetch the answer. Please try again.")
    else:
        st.warning("Please enter a question before clicking 'Get Answer'.")

