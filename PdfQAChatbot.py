import PyPDF2
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenLlama
from langchain.embeddings import LlamaEmbeddings
from langchain.vectorstores import FAISS

class PDFChatbot:
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
        """Sets up a QA chain using LangChain with OpenLLaMA."""
        embeddings = LlamaEmbeddings()
        doc_search = FAISS.from_texts([self.document_text], embeddings)

        return ConversationalRetrievalChain.from_llm(
            llm=ChatOpenLlama(model="openllama-7b"),
            retriever=doc_search.as_retriever()
        )

    def chat(self, question, history=[]):
        """Answers questions based on the PDF content."""
        response = self.qa_chain.run(question=question, chat_history=history)
        return response

if __name__ == "__main__":
    # Specify the path to your PDF
    pdf_path = "example.pdf"

    # Create a chatbot instance
    chatbot = PDFChatbot(pdf_path)

    print("Chatbot is ready! Ask questions about the PDF.")
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chatbot. Goodbye!")
            break

        response = chatbot.chat(user_input, chat_history)
        print(f"Chatbot: {response}")
