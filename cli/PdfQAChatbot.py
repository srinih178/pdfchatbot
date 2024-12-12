import PyPDF2
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS

llama_model_name = "llama3.2"
#llama_model_name = "llama2:7b-chat"
#llama_model_name = "llama3-chatqa"

class PdfQAChatbot:

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.document_text = self._extract_text_from_pdf()
        self.qa_chain = self._setup_qa_chain()

    def _extract_text_from_pdf(self):
        """Extracts text from the PDF file."""
        text = ""
        with open(self.pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text

    def _setup_qa_chain(self):
        print("Document Text:", self.document_text[:500])  # Print a snippet of the extracted text

        """Sets up a QA chain using LangChain with ChatOllama."""
        embeddings = OllamaEmbeddings(model=llama_model_name)  # Ensure correct configuration for embeddings
        doc_search = FAISS.from_texts([self.document_text], embeddings)

        print("QA Chain Initialized Successfully")
        return ConversationalRetrievalChain.from_llm(
            llm=ChatOllama(model=llama_model_name),  # Ensure the model name is valid
            retriever=doc_search.as_retriever()
        )

    def chat(self, question, history=[]):
        """Answers questions based on the PDF content."""
        response = self.qa_chain.run(question=question, chat_history=history)
        return response

if __name__ == "__main__":
    # Specify the path to your PDF
    pdf_path = "../restaurant_bill.pdf"

    # Create a chatbot instance
    chatbot = PdfQAChatbot(pdf_path)


    print("Chatbot is ready! Ask questions about the PDF.")
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Goodbye!")
            break

        response = chatbot.chat(user_input, chat_history)
        print(f"Chatbot: {response}")
