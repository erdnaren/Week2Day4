# Multi-PDF RAG Assistant

A Streamlit app that lets you upload multiple PDFs, index all their text in one FAISS vector database, and ask questions across the collection. Answers include inline citations such as `[annual_report.pdf, p. 12]` plus a de-duplicated source list.

## 1. Requirements

- Python 3.10 or newer (Python 3.11 recommended)
- An OpenAI API key with API access/billing enabled
- Internet access while installing packages and calling OpenAI

## 2. Create and activate a virtual environment (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

## 3. Configure the API key

Copy `.env.example` to `.env` and add your key:

```env
OPENAI_API_KEY=your_real_key
```

Do not commit `.env` to GitHub or share it. Alternatively, paste the key into the app's password field.

## 4. Run

From the project directory:

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).

## 5. Use the app

1. Enter your OpenAI API key in the sidebar.
2. Select multiple PDF files in the uploader. If your browser supports folder selection, select a folder; otherwise multi-select the PDFs.
3. Click **Build / rebuild knowledge base**.
4. Ask questions in the chat box.
5. Check the inline citations and expand **Sources used** to see the source filename and page number.

All PDFs in that indexing run are combined into one FAISS index. Rebuild after changing the selected files.

## 6. How it works

```text
Multiple PDFs
    |
    v
PyPDFLoader (extract text page by page)
    |
    v
RecursiveCharacterTextSplitter
    |  chunk_size=1000, overlap=150
    v
OpenAI text-embedding-3-small
    |
    v
One FAISS vector store
    |
Question --> MMR retriever --> relevant chunks
                              |
                              v
                     GPT-4o-mini + citations
                              |
                              v
                    Answer + file/page sources
```

Each chunk inherits its original filename and page metadata. PyPDFLoader page numbers are zero-based, so the app converts them to human-readable page numbers starting at 1.

## 7. Important limitations

- Scanned/image-only PDFs may have no extractable text. Add OCR (for example, Tesseract) if you need those documents.
- PDF page numbering can differ from printed page labels inside a document. Citations refer to the PDF's physical page order.
- Retrieval is not guaranteed to find every relevant passage. For critical use, verify the cited page in the original PDF.
- This starter app keeps the FAISS index in Streamlit session memory. Restarting the app or losing the session requires rebuilding it.
- The app sends extracted PDF text to OpenAI for embeddings and question answering. Do not upload confidential documents unless your organization's policies permit it.
- For production use, add authentication, file-size limits, persistent index storage, logging, and stronger document-level access controls.
