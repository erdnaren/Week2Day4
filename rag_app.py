
import streamlit as st
import os

from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="Conversational RAG Application",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Conversational RAG Application")
st.write("Upload PDF documents and ask multiple questions.")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found. Please configure your .env file.")
    st.stop()

# --------------------------------------------------
# 2. Initialize LLM and Embeddings
# --------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# --------------------------------------------------
# 3. Session State
# --------------------------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# 4. PDF Upload
# --------------------------------------------------

st.sidebar.header("Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if st.sidebar.button("Process Documents"):

    if uploaded_files:

        documents = []

        for pdf in uploaded_files:

            pdf_reader = PdfReader(pdf)

            for page_number, page in enumerate(pdf_reader.pages):

                text = page.extract_text()

                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": pdf.name,
                                "page": page_number + 1
                            }
                        )
                    )

        if not documents:
            st.sidebar.error("No readable text found in PDFs.")

        else:

            # Split documents
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(documents)

            # Create FAISS vector store
            st.session_state.vectorstore = (
                FAISS.from_documents(chunks, embeddings)
            )

            # Reset previous conversation when documents change
            st.session_state.chat_history = []
            st.session_state.messages = []

            st.sidebar.success(
                f"Successfully processed {len(chunks)} chunks!"
            )

    else:
        st.sidebar.warning("Please upload PDF files.")

# --------------------------------------------------
# 5. Create Conversational RAG Chain
# --------------------------------------------------

def create_rag_chain(vectorstore):

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    # Rewrite follow-up questions using chat history
    contextualize_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given the conversation history and latest question, "
            "rewrite the latest question as a standalone question. "
            "Do not answer it."
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_prompt
    )

    # Answer using retrieved documents
    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are a helpful AI assistant.

            Answer the question using the retrieved context.

            If the answer is not available in the context,
            clearly say that the information is not available
            in the uploaded documents.

            Context:
            {context}
            """
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(
        llm,
        qa_prompt
    )

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        question_answer_chain
    )

    return rag_chain

# --------------------------------------------------
# 6. Display Previous Conversations
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --------------------------------------------------
# 7. Multiple Questions
# --------------------------------------------------

if st.session_state.vectorstore is None:

    st.info("👈 Upload and process PDF documents to start chatting.")

else:

    rag_chain = create_rag_chain(
        st.session_state.vectorstore
    )

    question = st.chat_input(
        "Ask your question about the uploaded documents..."
    )

    if question:

        # Display user question
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        # Generate response
        with st.chat_message("assistant"):

            with st.spinner("Finding the answer..."):

                try:

                    response = rag_chain.invoke({
                        "input": question,
                        "chat_history": st.session_state.chat_history
                    })

                    answer = response["answer"]

                    st.markdown(answer)

                    # Update conversation memory
                    st.session_state.chat_history.extend([
                        HumanMessage(content=question),
                        AIMessage(content=answer)
                    ])

                    # Save conversation
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:

                    st.error(f"Error generating answer: {e}")

# --------------------------------------------------
# 8. Clear Chat
# --------------------------------------------------

if st.sidebar.button("🗑️ Clear Chat"):

    st.session_state.messages = []
    st.session_state.chat_history = []

    st.sidebar.success("Chat history cleared!")

    st.rerun()
