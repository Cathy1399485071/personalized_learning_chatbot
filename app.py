import os
import openai
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure your OpenAI API key
openai.api_key = 'OPENAI_API_KEY'
os.environ['OPENAI_API_KEY'] = 'OPENAI_API_KEY'

# Mock RAG Model Response
def generate_answer(query, topics):
    # Combine query with topics and return a mock answer
    return f"Mock answer for question: '{query}' with topics: {', '.join(topics)}"

selected_topics = []
dag_data = []
vectordb = None

def extract_topics_from_document(pages):
    for page in pages:
        document_text = page.page_content
    

    # Prompt to instruct GPT to extract key topics in a hierarchical format
    prompt = f"Extract key topics from the following document. Structure them as main topics and subtopics as needed:\n\n{document_text}\n\nReturn a list of topics in hierarchical format."

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    # Process response to get a list of topics
    topics = response.choices[0].message['content'].strip().split('\n')
    # print(topics)
    return [{"id": topic} for topic in topics if topic]  # Format as list of topics

def write_to_vectordb(pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_documents(pages)
    persist_directory = 'docs/chroma/'
    global vectordb
    if vectordb is not None:
        vectordb = None  # Dereference the instance
    if os.path.exists(persist_directory):
        try:
            shutil.rmtree(persist_directory)  # Delete the directory and its contents
            print(f"Successfully deleted ChromaDB directory: {persist_directory}")
        except Exception as e:
            print(f"Failed to delete ChromaDB directory {persist_directory}: {e}")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    vectordb.persist()
    # print(type(vectordb))
    print(vectordb._collection.count())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load and process the PDF
        loader = PyPDFLoader(filepath)
        pages = loader.load()
        write_to_vectordb(pages)
        global dag_data
        dag_data = extract_topics_from_document(pages)
        # print(dag_data)

        return jsonify({"status": "success", "topics": dag_data}), 200
    return jsonify({"status": "error"}), 400

@app.route('/select_topics', methods=['POST'])
def select_topics():
    global selected_topics
    selected_topics = request.json.get('topics', [])
    print(selected_topics)
    return jsonify({"status": "success"}), 200

@app.route('/query', methods=['POST'])
def query():
    select_topics_str = ", ".join(selected_topics)
    llm = ChatOpenAI(model_name='gpt-4', temperature=0)
    query_text = request.json.get('query', "")
    if query_text:
        qa_chain_mr = RetrievalQA.from_chain_type(
            llm,
            retriever=vectordb.as_retriever(),
            chain_type="stuff"
        )
        result = qa_chain_mr({
            "query": query_text,
            "learned_topics": select_topics_str
            })
        result["result"]
        response = result["result"]
        return jsonify({"answer": response}), 200
    return jsonify({"status": "error"}), 400

if __name__ == '__main__':
    app.run(port=5001, debug=True)
