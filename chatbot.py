# Load necessary modules
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
import streamlit as st
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure the Gemini API with the key from .env file
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load Gemini Pro model and start the chat session
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

# Define the function to get responses from the Gemini model
def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

# Define a PromptTemplate with a sample prompt structure
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""
You are an intelligent assistant using Gemini Pro. Your role is to answer user questions accurately and efficiently.

Question: {question}
Answer:"""
)

# Wrap the function to fetch Gemini responses as a RunnableLambda
llm_runnable = RunnableLambda(
    lambda input: get_gemini_response(input.to_string())
)

# Create a pipeline sequence combining the prompt and the LLM
pipeline = RunnableSequence(prompt_template, llm_runnable)

# Initialize the Streamlit app
st.set_page_config(page_title="Gemeni AI Using Langchain")
st.header("Gemini LLM Application with LangChain")

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Get user input from Streamlit interface
user_input = st.text_input("Input: ", key="input")
submit = st.button("Ask the question")

if submit and user_input:
    # Use the LangChain pipeline to get the response
    result = pipeline.invoke({"question": user_input})

    # Add user query and response to session state chat history
    st.session_state['chat_history'].append(("You", user_input))
    st.subheader("The Response is")

    # Display response chunk by chunk
    for chunk in result:
        st.write(chunk.text)
        st.session_state['chat_history'].append(("Bot", chunk.text))

# Display chat history
st.subheader("The Chat History is")
for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")
