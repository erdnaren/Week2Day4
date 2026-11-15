
import os
import hashlib
from io import BytesIO
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="Multi-PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Multi-PDF RAG Assistant")
st.caption(
    "Upload multiple PDFs and ask questions across all "
    "your documents."
)

if not os.getenv("OPENAI_API_KEY"):
    st.error("Please configure OPENAI_API_KEY in your .env file.")
    st.stop()


# --------------------------------------------------
# 2. INITIALIZE MODELS
# --------------------------------------------------

@st.cache_resource
def initialize_models():

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0
    )

    return embeddings, llm


embeddings, llm = initialize_models()


# --------------------------------------------------
# 3. EXTRACT TEXT FROM MULTIPLE PDFS
# --------------------------------------------------

def extract_pdf_documents(uploaded_files):

    documents = []
    errors = []

    for uploaded_file in uploaded_files:

        try:
            pdf_bytes = uploaded_file.getvalue()
            reader = PdfReader(BytesIO(pdf_bytes))

            for page_number, page in enumerate(
                reader.pages, start=1
            ):

                text = page.extract_text() or ""

                if not text.strip():
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": uploaded_file.name,
                            "page": page_number
                        }
                    )
                )

        except Exception as error:
            errors.append(
                f"{uploaded_file.name}: {str(error)}"
            )

    return documents, errors


# --------------------------------------------------
# 4. SPLIT DOCUMENTS INTO CHUNKS
# --------------------------------------------------

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    return chunks


# --------------------------------------------------
# 5. CREATE ONE FAISS VECTOR DATABASE
# --------------------------------------------------

def create_vector_database(chunks):

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore


# --------------------------------------------------
# 6. QUESTION ANSWERING
# --------------------------------------------------

def answer_question(question, vectorstore):

    # Retrieve relevant chunks from ALL PDFs
    retrieved_docs = vectorstore.similarity_search(
        question,
        k=6
    )

    if not retrieved_docs:
        return (
            "I couldn't find relevant information in "
            "the uploaded documents.",
            []
        )

    # Build context with source identifiers
    context_parts = []

    for index, doc in enumerate(retrieved_docs, start=1):

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        context_parts.append(
            f"""
[SOURCE {index}]
File: {source}
Page: {page}

Content:
{doc.page_content}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a precise document question-answering assistant.

Answer the user's question using ONLY the supplied context.

Rules:
1. Do not invent facts or information.
2. If the answer is not available in the context,
   clearly say that the documents do not contain
   enough information.
3. Cite every factual claim using the source identifier.
4. Use citations in this format: [SOURCE 1].
5. When multiple sources support an answer, cite all
   relevant sources.
6. Distinguish conflicting information between documents.
7. Never fabricate a filename or page number.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = llm.invoke(prompt)

    return response.content, retrieved_docs


# --------------------------------------------------
# 7. SESSION STATE
# --------------------------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "file_signature" not in st.session_state:
    st.session_state.file_signature = None


# --------------------------------------------------
# 8. SIDEBAR - PDF UPLOAD
# --------------------------------------------------

with st.sidebar:

    st.header("📂 Document Upload")

    uploaded_files = st.file_uploader(
        "Upload your PDF documents",
        type=["pdf"],
        accept_multiple_files="directory",
        help="Select a folder containing PDFs, or upload "
             "multiple PDFs."
    )

    # Also allow selecting multiple PDFs individually.
    # Streamlit's directory uploader accepts folders.
    st.caption("You can select a folder or multiple PDF files.")

    if st.button(
        "Process Documents",
        type="primary",
        use_container_width=True
    ):

        if not uploaded_files:
            st.warning("Please upload at least one PDF.")

        else:

            with st.spinner(
                "Extracting, chunking and indexing documents..."
            ):

                documents, errors = extract_pdf_documents(
                    uploaded_files
                )

                if errors:
                    for error in errors:
                        st.warning(error)

                if not documents:
                    st.error(
                        "No readable PDF text was found."
                    )

                else:

                    chunks = split_documents(documents)

                    vectorstore = create_vector_database(
                        chunks
                    )

                    st.session_state.vectorstore = vectorstore

                    st.session_state.processed_files = list(
                        dict.fromkeys(
                            doc.metadata["source"]
                            for doc in documents
                        )
                    )

                    st.session_state.file_signature = hashlib.sha256(
                        b"".join(
                            sorted(
                                f.getvalue()
                                for f in uploaded_files
                            )
                        )
                    ).hexdigest()

                    st.success("Documents indexed successfully!")

                    st.info(
                        f"""
                        Files processed: {
                            len(st.session_state.processed_files)
                        }

                        Pages extracted: {len(documents)}

                        Chunks created: {len(chunks)}
                        """
                    )

    st.divider()

    st.subheader("📑 Indexed Documents")

    if st.session_state.processed_files:

        for filename in st.session_state.processed_files:
            st.write(f"📄 {filename}")

    else:
        st.caption("No documents indexed yet.")


# --------------------------------------------------
# 9. MAIN CHAT INTERFACE
# --------------------------------------------------

st.subheader("💬 Ask Questions Across Your PDFs")

if st.session_state.vectorstore is None:

    st.info(
        "Upload your PDFs in the sidebar and click "
        "'Process Documents' to get started."
    )

else:

    question = st.text_input(
        "Enter your question",
        placeholder="What are the key findings across these PDFs?"
    )

    if st.button("Get Answer", type="primary"):

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            with st.spinner("Searching documents and generating answer..."):

                answer, sources = answer_question(
                    question,
                    st.session_state.vectorstore
                )

            st.markdown("### 📝 Answer")
            st.markdown(answer)

            st.divider()

            st.markdown("### 📚 Sources and References")

            seen = set()

            for index, doc in enumerate(sources, start=1):

                filename = doc.metadata.get(
                    "source", "Unknown"
                )

                page = doc.metadata.get(
                    "page", "Unknown"
                )

                source_key = (filename, page, doc.page_content)

                if source_key in seen:
                    continue

                seen.add(source_key)

                with st.expander(
                    f"📄 {filename} — Page {page}"
                ):

                    st.markdown(
                        f"**Source ID:** SOURCE {index}"
                    )

                    st.markdown("**Retrieved content:**")

                    st.write(doc.page_content)

