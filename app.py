from flask import Flask, render_template, request, jsonify
import os
import requests
from werkzeug.utils import secure_filename
from chroma_tasks import load_chroma_documents, clear_chroma_database
from query_tasks import query_rag

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'D:/LLM/Data'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Ensure upload dir exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

LLAMA_SERVER_URL = "http://localhost:8080/completion"  # llama.cpp REST server URL

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# --- Stubs for required functions ---
def search_rag(question):
    return f"🔍 Context retrieved for: {question}\n(This would be actual RAG results.)"

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/init_db', methods=['POST'])
def init_db():
    try:
        clear_chroma_database()
        print("Database cleared ✅")
        return jsonify(success=True, message="DB iniciada!")
    except Exception as e:
        return jsonify(success=False, message=str(e))

@app.route('/upload_pdfs', methods=['POST'])
def upload_pdfs():
    try:
        if 'files' not in request.files:
            return jsonify(success=False, message="No se enviaron archivos.")
        
        files = request.files.getlist('files')
        saved_files = []

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                saved_files.append(filename)

        if saved_files:
            load_chroma_documents()
            return jsonify(success=True, message=f"Se cargaron {len(saved_files)} PDFs", files=saved_files)
        else:
            return jsonify(success=False, message="No se cargaron PDFs.")
    except Exception as e:
        return jsonify(success=False, message=str(e))

@app.route('/ask_question', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data.get("question", "")
        use_rag = data.get("use_rag", False)

        if not question:
            return jsonify(success=False, message="No se capturó una pregunta.")

        # If RAG is checked, preappend context
        if use_rag:
            prompt = query_rag(question)
            print(f"RAG Context: {prompt}")
        else:
            prompt = question

        # Send request to llama.cpp server
        response = requests.post(
            LLAMA_SERVER_URL,
            json={
                "prompt": prompt,
                "n_predict": 256,
                "temperature": 0.7
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            answer = result.get("content", "⚠️ No se recibió respuesta.")
            return jsonify(success=True, answer=answer)
        else:
            return jsonify(success=False, message=f"LLAMA Server error {response.status_code}")
    except Exception as e:
        return jsonify(success=False, message=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True, port=5000)