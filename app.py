import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from gpt4all import GPT4All

st.set_page_config(
    page_title="AI Knowledge Base Bot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Company Knowledge Base Bot")

st.sidebar.header("Settings")

top_k = st.sidebar.slider(
    "Top K Results",
    min_value=1,
    max_value=10,
    value=3
)

chunk_size = st.sidebar.slider(
    "Chunk Size",
    min_value=200,
    max_value=1000,
    value=500
)

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_files = st.file_uploader(
    "Upload Company Documents",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    all_docs = []

    with st.spinner("Processing Documents..."):

        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                pdf_path = tmp_file.name

            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["document_name"] = uploaded_file.name

            all_docs.extend(docs)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(all_docs)

        if not chunks:
            st.error("❌ No chunks created. Try reducing chunk size or upload a larger document.")
        else:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            vectorstore = FAISS.from_documents(chunks, embeddings)

            retriever = vectorstore.as_retriever(
                search_kwargs={"k": top_k}
            )

            st.success(f"✅ {len(uploaded_files)} Documents Processed")

            user_question = st.chat_input("Ask about company documents...")

            if user_question:
                st.session_state.messages.append({"role": "user", "content": user_question})

                retrieved_docs = retriever.invoke(user_question)

                context = "\n".join([doc.page_content for doc in retrieved_docs])

                prompt = PromptTemplate(
                    input_variables=["context", "question"],
                    template="""
You are a company knowledge assistant.

Answer only from the provided context.

If the answer is not found in the context,
say:
"I couldn't find that information in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
"""
                )

                final_prompt = prompt.format(context=context, question=user_question)

                llm = GPT4All("mistral-7b-instruct-v0.1.Q4_0.gguf")
                answer = llm.generate(final_prompt)

                st.session_state.messages.append({"role": "assistant", "content": answer})

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

            if user_question:
                st.subheader("📄 Source Documents")
                shown = set()
                for doc in retrieved_docs:
                    doc_name = doc.metadata.get("document_name", "Unknown")
                    page_no = doc.metadata.get("page", 0) + 1
                    key = (doc_name, page_no)
                    if key not in shown:
                        shown.add(key)
                        st.write(f"📄 {doc_name} (Page {page_no})")
