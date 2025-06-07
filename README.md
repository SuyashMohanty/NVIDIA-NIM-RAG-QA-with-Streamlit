# 🚀 NVIDIA NIM RAG Q&A with Streamlit

This project demonstrates a powerful Retrieval Augmented Generation (RAG) application that allows users to ask questions about their own documents. It leverages **NVIDIA NIM** for both high-performance LLM inference (`meta/llama3-70b-instruct`) and embedding generation. The application is built with **Langchain** and features an interactive web interface created with **Streamlit**.

---
## ✨ Features

* **NVIDIA NIM Integration**: Utilizes NVIDIA's Inference Microservices (NIM) for accessing state-of-the-art models.
* **Advanced LLM**: Employs the `meta/llama3-70b-instruct` model for generating accurate and context-aware answers.
* **High-Performance Embeddings**: Uses `NVIDIAEmbeddings` for efficiently converting documents into vector representations.
* **Document Q&A**: Allows you to chat with your own PDF documents. The app loads PDFs from a specified directory (`./us_census`).
* **Efficient In-Memory Search**: Uses **FAISS** (Facebook AI Similarity Search) as a local vector store for fast and efficient document retrieval.
* **Interactive UI**: A user-friendly web interface built with **Streamlit** that allows for easy document processing and querying.
* **Langchain Powered**: The entire RAG pipeline (loading, splitting, embedding, retrieving, and generating) is orchestrated using the Langchain framework.

---
## ⚙️ How it Works

The application follows a classic RAG pipeline:

1.  **Environment Setup**: The application loads the required NVIDIA API key from environment variables.
2.  **Streamlit Interface**: A Streamlit UI is launched, providing a title, a button to start the embedding process, and an input box for user questions.
3.  **Document Loading & Processing**: When the user clicks the **"Documents Embedding"** button:
    * PDF files are loaded from the `./us_census` directory using `PyPDFDirectoryLoader`.
    * The documents are split into smaller, manageable chunks using `RecursiveCharacterTextSplitter`.
    * The `NVIDIAEmbeddings` model is used to convert these text chunks into vector embeddings.
    * These embeddings are stored in a FAISS vector store, which is saved in the Streamlit session state for quick access.
4.  **Question Answering**:
    * When a user types a question into the input box:
    * The FAISS vector store is used as a **retriever** to find the most relevant document chunks based on the user's query.
    * A Langchain "stuff" chain (`create_stuff_documents_chain`) takes the retrieved documents and the user's question and formats them into a prompt based on a defined template.
    * This prompt is sent to the `ChatNVIDIA` LLM (`meta/llama3-70b-instruct`).
    * The LLM generates a response based *only* on the provided context from the documents.
    * The final answer is displayed in the Streamlit interface.

---
## 📋 Requirements

* Python 3.x
* An NVIDIA API Key with access to NIM.
* Required Python libraries: `streamlit`, `langchain`, `langchain-nvidia-ai-endpoints`, `langchain-community`, `pypdf`, `faiss-cpu`, `python-dotenv`.

---
## 🛠️ Setup & Installation

1.  **Clone the repository or download the `finalapp.py` script.**

2.  **Create a Directory for Documents**:
    * In the same directory as `finalapp.py`, create a folder named `us_census`.
    * Place all the PDF documents you want to query inside this `us_census` folder.

3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

4.  **Install the required Python libraries:**
    ```bash
    pip install streamlit langchain langchain-nvidia-ai-endpoints langchain-community langchain-text-splitters langchain-core pypdf faiss-cpu python-dotenv
    ```

5.  **Set up NVIDIA API Key:**
    * Obtain an API key from the [NVIDIA AI Playground](https://build.nvidia.com/).
    * Create a `.env` file in the root directory of your project.
    * Add your NVIDIA API key to the `.env` file. **Note**: The script loads the key with the name `NVEDIA_API_KEY` (a typo from the source file).
        ```env
        NVEDIA_API_KEY="your_nvidia_api_key_here"
        ```
    The script uses `load_dotenv()` to load this key automatically.

---
## ▶️ Usage

1.  **Run the Streamlit App**:
    * Open your terminal in the project's root directory.
    * Execute the following command:
        ```bash
        streamlit run finalapp.py
        ```
    * Your web browser should open with the application's UI.

2.  **Embed Your Documents**:
    * The first time you run the app, or whenever you change the documents in the `us_census` folder, you must first click the **"Documents Embedding"** button.
    * Wait for the "Vector Store DB Is Ready using Nvidia Embeddings" message to appear. This indicates that your documents have been processed and are ready for querying.

3.  **Ask Questions**:
    * Once the vector store is ready, type your question into the text input box labeled "Enter Your Questions From Documents".
    * Press Enter. The application will process your query and display the answer generated by the RAG pipeline.

---
