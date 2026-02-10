import os
from dotenv import load_dotenv 
from flask import Flask, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import ConversationalRetrievalChain 
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv(override=True)

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("CRITICAL ERROR: GOOGLE_API_KEY is missing from .env file.")

app = Flask(__name__)

session_memories = {}
DB_DIRECTORY = "./chroma_db"

MODEL_NAME = "gemini-2.5-flash"

def getGeminiLLM():
    """
    Returns the LLM instance with specific parameters.
    Temperature 0.3 reduces hallucinations for factual Q&A.
    """
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3, 
        max_output_tokens=8192
    )

def getEmbeddings():
    """
    Returns the Google GenAI Embedding model.
    Optimized for working with Gemini LLMs.
    """
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

def preprocessDocs(docs):
    """
    Splits documents into 2000-character chunks with overlap to maintain context.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)

@app.route('/update_course', methods=['POST'])
def updateCourse():
    try:
        courseId = request.form['courseId']
        pdfPaths = request.form['pdfPaths'].split(',')
        
        print(f"--- Processing Update for Course: {courseId} ---")
        
        all_docs = []
        for path in pdfPaths:
            path = path.strip()
            if not os.path.exists(path):
                print(f"Warning: File not found {path}")
                continue
                
            print(f"Loading: {path}")
            loader = PyPDFLoader(path)
            all_docs.extend(loader.load())
            
        if not all_docs:
            return jsonify({"status": "error", "message": "No valid documents found."}), 400

        # Chunking
        print(f"Chunking {len(all_docs)} pages...")
        chunks = preprocessDocs(all_docs)
        
        # Persist to Vector DB
        print("Embedding and Indexing...")
        Chroma.from_documents(
            documents=chunks,
            embedding=getEmbeddings(),
            persist_directory=DB_DIRECTORY,
            collection_name=f"course_{courseId}" # ISOLATION: Unique collection per course
        )
        
        # Clear old memory for this course to start fresh
        if courseId in session_memories:
            del session_memories[courseId]
            
        print("Success!")
        return jsonify({
            "status": "success", 
            "message": f"Course {courseId} indexed successfully with {len(chunks)} chunks."
        }), 200
        
    except Exception as e:
        print(f"Update Error: {e}") 
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/ask', methods=['POST'])
def askQuestion():
    try:
        data = request.get_json()
        courseId = data.get('courseId')
        question = data.get('question')

        if not courseId or not question:
            return jsonify({"status": "error", "message": "Missing courseId or question"}), 400

        # 1. Load the specific course Knowledge Base
        vectorstore = Chroma(
            persist_directory=DB_DIRECTORY,
            embedding_function=getEmbeddings(),
            collection_name=f"course_{courseId}"
        )
        
        # 2. Get or Create Memory
        if courseId not in session_memories:
            session_memories[courseId] = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True,
                output_key='answer'
            )

        # 3. Build RAG Chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=getGeminiLLM(), 
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), # Top 5 relevant chunks
            memory=session_memories[courseId],
            verbose=False
        )
    
        print(f"Asking ({courseId}): {question}")
        result = chain.invoke({"question": question})
        
        return jsonify({"answer": result["answer"]})
        
    except Exception as e:
        print(f"Ask Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return "<h1>Gemini RAG API is Online </h1>"

if __name__ == '__main__':
    # Use the PORT environment variable if it's available (needed for Render)
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port)