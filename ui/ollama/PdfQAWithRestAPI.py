from flask import Flask, request, jsonify
import pdfplumber
import requests
import json
# Initialize Flask app
app = Flask(__name__)

# Global variable to store text chunks
text_chunks = []

# Ollama API configuration
OLLAMA_API_URL = "http://localhost:11434/api"
OLLAMA_MODEL_NAME = "llama3.2"  # Replace with the model you are using


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


def chunk_text(text, max_chars=500):
    """Split text into smaller, manageable chunks."""
    chunks = []
    while len(text) > max_chars:
        split_idx = text[:max_chars].rfind(" ")
        chunks.append(text[:split_idx])
        text = text[split_idx:]
    chunks.append(text)
    return chunks


def call_ollama_api(context, question):
    """Interact with the Ollama model using REST API."""
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    url = f"{OLLAMA_API_URL}/generate"
    payload = {"model": OLLAMA_MODEL_NAME, "prompt": prompt}
    headers = {"Content-Type": "application/json"}
    print("payload:", payload)
    print("url:", url)
    response = requests.post(url, json=payload, headers=headers)
    print("response:", response.text, response.status_code)
    if response.status_code == 200:
        #print("json", response.json())
        generated_text = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_line = json.loads(line.decode("utf-8"))  # Parse each line as JSON
                    generated_text += json_line.get("response", "")  # Append the "response" field
                except json.JSONDecodeError as e:
                    print(f"Failed to parse line: {line}. Error: {e}")
        print("generated_text", generated_text)
        return generated_text
        #return response.json().get("text", "No answer generated.")
    else:
        print("Error:", {response.status_code})
        raise Exception(f"Ollama API call failed: {response.status_code}, {response.text}")


@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """Endpoint to upload and process a PDF."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the file and extract text
    file_path = f"./{file.filename}"
    file.save(file_path)
    text = extract_text_from_pdf(file_path)

    global text_chunks
    text_chunks = chunk_text(text)
    return jsonify({"message": "PDF uploaded and processed successfully!"})


@app.route('/ask', methods=['POST'])
def ask_question():
    """Endpoint to ask a question based on the uploaded PDF."""
    if not text_chunks:
        return jsonify({"error": "No PDF has been uploaded and processed"}), 400

    data = request.json
    if 'question' not in data:
        return jsonify({"error": "No question provided"}), 400

    question = data['question']
    #context = " ".join(text_chunks[:30])  # Use the first few chunks for context
    context = " ".join(text_chunks)  # Use all chunks for context
    try:
        print("context", context)
        answer = call_ollama_api(context, question)
        print("answer:", answer)
        return jsonify({"question": question, "answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
