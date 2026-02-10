import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader

# -------------------------------------------------------------------
# Environment setup
# -------------------------------------------------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY is not set")

# -------------------------------------------------------------------
# Flask App
# -------------------------------------------------------------------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return {"status": "ok", "service": "Gemini RAG API"}

# -------------------------------------------------------------------
# Globals
# -------------------------------------------------------------------
session_memories = {}
DB_DIRECTORY = "./chroma_db"

MODEL_NAME = "gemini-1.5-flash"  # SAFE & STABLE

# -------------------------------------------------------------------
# LLM & Embeddings
# -------------------------------------------------------------------
def getGeminiLLM():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3,
        max_output_tokens=8192
    )


def getEmbeddings():
    return GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",  
        google_api_key=GOOGLE_API_KEY
    )


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------
def preprocessDocs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.route("/update_course", methods=["POST"])
def update_course():
    try:
        course_id = request.form.get("courseId")
        pdf_paths = request.form.get("pdfPaths", "").split(",")

        if not course_id:
            return jsonify({"error": "courseId is required"}), 400

        all_docs = []

        for path in pdf_paths:
            path = path.strip()
            if not path or not os.path.exists(path):
                continue

            loader = PyPDFLoader(path)
            all_docs.extend(loader.load())

        if not all_docs:
            return jsonify({"error": "No valid PDFs found"}), 400

        chunks = preprocessDocs(all_docs)

        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=getEmbeddings(),
            persist_directory=DB_DIRECTORY,
            collection_name=f"course_{course_id}"
        )
        vectordb.persist()

        session_memories.pop(course_id, None)

       return jsonify({
    "status": "success", 
    "message": f"Course {course_id} indexed successfully with {len(chunks)} chunks."
})

    except Exception as e:
        print("Update error:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(force=True)
        course_id = data.get("courseId")
        question = data.get("question")

        if not course_id or not question:
            return jsonify({"error": "courseId and question required"}), 400

        vectordb = Chroma(
            persist_directory=DB_DIRECTORY,
            embedding_function=getEmbeddings(),
            collection_name=f"course_{course_id}"
        )

        if course_id not in session_memories:
            session_memories[course_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )

        chain = ConversationalRetrievalChain.from_llm(
            llm=getGeminiLLM(),
            retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
            memory=session_memories[course_id],
            verbose=False
        )

        result = chain.invoke({"question": question})
        return jsonify({
            "status": "success",
            "message": f"Course {course_id} indexed successfully with {len(chunks)} chunks."
        })

    except Exception as e:
        print("Update error:", e)
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------------------------
# Local Run (ignored by Gunicorn)
# -------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
