import PyPDF2
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS

llama_model_name = "llama2:7b-chat"

class PdfQAChatbotMulti:
    def __init__(self, pdf_paths):
        self.pdf_paths = pdf_paths
        self.document_texts = self._extract_texts_from_pdfs()
        self.qa_chain = self._setup_qa_chain()

    def _extract_texts_from_pdfs(self):
        """Extracts text from multiple PDF files."""
        texts = []
        for pdf_path in self.pdf_paths:
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text()
            texts.append(text)
        return texts

    def _setup_qa_chain(self):
        """Sets up a QA chain using LangChain with ChatOllama."""
        print("Document Text:", self.document_texts[:500])  # Print a snippet of the extracted text
        embeddings = OllamaEmbeddings(model=llama_model_name)
        doc_search = FAISS.from_texts(self.document_texts, embeddings)

        return ConversationalRetrievalChain.from_llm(
            llm=ChatOllama(model=llama_model_name),
            retriever=doc_search.as_retriever()
        )

    def chat(self, question, history=[]):
        """Answers questions based on the PDF content."""
        inputs = {
            "question": question,
            "chat_history": history
        }
        return self.qa_chain.invoke(inputs)

if __name__ == "__main__":
    # Specify the paths to your PDFs
    pdf_paths = ["restaurant_bill.pdf", "tdr1.pdf"]

    # Create a chatbot instance
    chatbot = PdfQAChatbotMulti(pdf_paths)

    print("Chatbot is ready! Ask questions about the PDFs.")
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Goodbye!")
            break

        response = chatbot.chat(user_input, chat_history)
        print(f"Chatbot: {response}")
