import streamlit as st
import warnings
import dotenv
import os
import time
import shutil
import hashlib
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up the Streamlit app
st.set_page_config(page_title="Fraud Suraksha - Scam Detection Assistant", page_icon="🛡️")

# App title and description
st.title("🛡️ Fraud Suraksha")
st.subheader("AI-powered Fraud Detection Assistant")
st.markdown("""
This app helps you verify if a message, person, or situation might be fraudulent.
Simply describe the situation, and our AI will analyze it for potential fraud signals.
""")

# Load environment variables
dotenv.load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

def get_pdf_hash(pdf_path):
    """Calculate hash of PDF file to detect changes."""
    with open(pdf_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def needs_database_update():
    """Check if the database needs to be updated."""
    chroma_dir = "chroma_db"
    pdf_path = Path("GMaps.pdf")
    
    # If database doesn't exist, it needs to be created
    if not os.path.exists(chroma_dir):
        return True
        
    # If PDF doesn't exist, we can't check for updates
    if not pdf_path.exists():
        return False
        
    # Check if PDF hash has changed
    current_hash = get_pdf_hash(pdf_path)
    hash_file = Path(chroma_dir) / "pdf_hash.txt"
    
    if not hash_file.exists():
        return True
        
    with open(hash_file, 'r') as f:
        stored_hash = f.read().strip()
        
    return current_hash != stored_hash

def update_database():
    """Update the vector database with current PDF content."""
    try:
        # Initialize embeddings and model
        gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Load the PDF using relative path
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        pdf_path = current_dir / "GMaps.pdf"
        
        if not pdf_path.exists():
            # Fallback to alternative locations
            potential_paths = [
                Path("./datafiles/GMaps.pdf"),  # Current directory
                Path("Google Maps Scams.pdf"),  # Alternative filename
            ]
            
            for path in potential_paths:
                if path.exists():
                    pdf_path = path
                    break
            else:
                raise FileNotFoundError("Could not find the PDF file with scam information")
        
        loader = PyPDFLoader(str(pdf_path))
        doc = loader.load()
        
        # Split the document
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(doc)
        
        # Create vector store and retriever
        chroma_dir = "chroma_db"
        if os.path.exists(chroma_dir):
            shutil.rmtree(chroma_dir)
            
        os.makedirs(chroma_dir, exist_ok=True)
            
        # Create the vector store with explicit persistent directory
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=gemini_embeddings, 
            persist_directory=chroma_dir
        )
        
        # Save the current PDF hash
        with open(Path(chroma_dir) / "pdf_hash.txt", 'w') as f:
            f.write(get_pdf_hash(pdf_path))
            
        return vectorstore.as_retriever()
        
    except Exception as e:
        raise Exception(f"Failed to update database: {str(e)}")

# Initialize message store for chat history
def get_session_history(session_id):
    if "message_store" not in st.session_state:
        st.session_state.message_store = {}
    
    if session_id not in st.session_state.message_store:
        st.session_state.message_store[session_id] = ChatMessageHistory()
        # Initialize with system message using the correct method
        st.session_state.message_store[session_id].add_message(
            SystemMessage(content="You are Fraud Suraksha, an AI assistant that helps users verify if messages, people, or situations might be fraudulent. "
            "Remember details from our conversation, especially the user's name if they share it.")
        )
        
    return st.session_state.message_store[session_id]

# Initialize session
if "session_id" not in st.session_state:
    st.session_state.session_id = "user_session_" + str(hash(str(st.session_state)))

# Display chat messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm Fraud Suraksha, your AI-powered fraud detection assistant. Describe a suspicious situation, message, or person, and I'll help you assess if it might be a scam."}
    ]
    
    # Initialize the chat history with the welcome message
    chat_history = get_session_history(st.session_state.session_id)
    chat_history.add_message(
        AIMessage(content="Hello! I'm Fraud Suraksha, your AI-powered fraud detection assistant. Describe a suspicious situation, message, or person, and I'll help you assess if it might be a scam.")
    )

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Add sidebar with information
with st.sidebar:
    st.header("About Fraud Suraksha")
    st.markdown("""
    ### How to use this app:
    
    1. Describe a suspicious message, call, or situation
    2. Our AI will analyze it against known scam patterns
    3. Get a risk assessment and recommended actions
    
    ### Common fraud types we can detect:
    - Google Maps scams
    - Fake reviews/ratings requests
    - Phishing attempts
    - Advanced fee fraud
    - Impersonation scams
    
    ### Disclaimer
    This tool provides analysis based on known patterns but cannot guarantee detection of all scams. Always exercise caution with suspicious communications.
    """)
    
    # Add debug section to show chat history (helpful for debugging)
    st.expander("Debug: Chat History", expanded=False).write(
        str(st.session_state.get("message_store", {}).get(st.session_state.get("session_id", ""), "No messages yet"))
    )

# Initialize RAG chain with proper error handling
@st.cache_resource
def initialize_rag_chain():
    try:
        # Initialize embeddings and model
        gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            convert_system_message_to_human=True,  # Important: Gemini doesn't support system messages directly
            temperature=0.7,
            max_output_tokens=2048,  # Increased token limit for more detailed responses
        )
        
        # Check if database needs updating
        if needs_database_update():
            retriever = update_database()
        else:
            # Load existing database
            chroma_dir = "chroma_db"
            vectorstore = Chroma(
                persist_directory=chroma_dir,
                embedding_function=gemini_embeddings
            )
            retriever = vectorstore.as_retriever()
        
        # Define the system prompt to include in the retrieval context
        system_context = (
            "You are an AI-powered fraud detection assistant designed to help users verify whether a person, message, or situation is fraudulent or safe. "
            "Use the following retrieved context to analyze scam patterns and assess the fraud risk score. "
            "If you don't have enough data to confirm, suggest caution and advise the user to be careful. "
            "Ask relevant follow-up questions if needed to gather more details before making a decision. "
            "Provide structured responses, including fraud risk score, detected red flags, and recommended actions. "
            "Never generate fake reports, financial guarantees, or unverifiable claims. "
            "Remember personal details that users share with you, such as their name."
            "\n\n"
            "{context}"
        )
        
        # Create chat prompt template with history placeholder
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_context),
                MessagesPlaceholder(variable_name="chat_history"),  # Include chat history
                ("human", "{input}"),
            ]
        )
        
        # Create the question answering chain
        question_answering_chain = create_stuff_documents_chain(model, chat_prompt)
        
        # Create the retrieval chain
        rag_chain = create_retrieval_chain(retriever, question_answering_chain)
        
        # Set up conversation memory with the correct configuration
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",  # Key for input in the chain
            history_messages_key="chat_history",  # Key for history in the prompt
            output_messages_key="answer",  # Key for output in the chain response
        )
        
        return conversational_rag_chain, None
    
    except Exception as e:
        return None, str(e)

# Initialize RAG chain or display error
rag_chain_result, error_message = initialize_rag_chain()

if error_message:
    st.error(f"⚠️ Error initializing the AI assistant: {error_message}")
    st.info("The application is running in fallback mode. Some features may be limited.")

# Chat input processing with error handling
if prompt := st.chat_input("What suspicious activity would you like me to evaluate?"):
    # Add user message to chat history for UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get AI response with error handling
    with st.chat_message("assistant"):
        if not rag_chain_result:
            # Fallback mode - provide a basic response when AI is not available
            fallback_response = (
                "I'm currently having trouble connecting to my knowledge base. "
                "Here are some general tips about fraud detection:\n\n"
                "- Be wary of unexpected communications asking for personal information\n"
                "- Don't click on suspicious links or download attachments from unknown sources\n"
                "- Verify the identity of any business or person through official channels\n"
                "- Be suspicious of requests for payment through gift cards or wire transfers\n"
                "- Take time to research before making decisions - scammers often create urgency\n\n"
                "Please try again later when our connection to the AI service is restored."
            )
            st.write(fallback_response)
            
            # Add to UI message history
            st.session_state.messages.append({"role": "assistant", "content": fallback_response})
            
            # Also add to LangChain message history
            chat_history = get_session_history(st.session_state.session_id)
            chat_history.add_user_message(prompt)
            chat_history.add_ai_message(fallback_response)
        else:
            try:
                with st.spinner("Analyzing..."):
                    max_retries = 3
                    retry_count = 0
                    
                    # Get or create session history
                    chat_history = get_session_history(st.session_state.session_id)
                    
                    while retry_count < max_retries:
                        try:
                            # Invoke the chain with history
                            response = rag_chain_result.invoke(
                                {"input": prompt},
                                config={
                                    "configurable": {
                                        "session_id": st.session_state.session_id
                                    }
                                },
                            )
                            
                            # Get the answer
                            ai_response = response["answer"]
                            break
                            
                        except Exception as e:
                            retry_count += 1
                            if retry_count >= max_retries:
                                raise
                            time.sleep(1)  # Short delay before retrying
                    
                    # Display the response
                    st.write(ai_response)
                    
                    # Add response to UI message history
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    
            except Exception as e:
                error_response = (
                    f"I apologize, but I'm experiencing technical difficulties at the moment. "
                    f"Please try again with your question. If the problem persists, here are some general "
                    f"guidelines about spotting fraud:\n\n"
                    f"- Be cautious with unexpected messages or calls\n"
                    f"- Verify identities through official channels\n"
                    f"- Don't share personal information unless you're certain it's legitimate\n"
                    f"- Be wary of offers that seem too good to be true"
                )
                st.write(error_response)
                
                # Add to both history stores
                st.session_state.messages.append({"role": "assistant", "content": error_response})
                chat_history = get_session_history(st.session_state.session_id)
                chat_history.add_user_message(prompt)
                chat_history.add_ai_message(error_response)
                
                # Display the error
                st.error(f"Error: {str(e)}", icon="🚨")
