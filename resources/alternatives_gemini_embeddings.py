"""
RAG Implementation Summary
- Embedding options (Google AI and alternatives)
- LLM options (Google AI and alternatives)
- K-value calculations
"""

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Original Google AI Setup
def setup_google_ai():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        google_api_key=GOOGLE_API_KEY
    )

    return embeddings, llm

# Alternative 1: HuggingFace Setup
def setup_huggingface():
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.llms import HuggingFaceHub

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.7}
    )

    return embeddings, llm

# Best performing embedding models
from langchain.embeddings import HuggingFaceEmbeddings

# Option 1: All-MiniLM-L6-v2 (Best balance of speed and performance)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Option 2: MPNet (Better performance but slower)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

from langchain.llms import HuggingFaceHub

# Option 1: Mixtral-8x7B (Currently best open source model)
llm = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.7, "max_length": 4096}
)

# Option 2: Llama-2-70b-chat (Very good performance)
llm = HuggingFaceHub(
    repo_id="meta-llama/Llama-2-70b-chat-hf",
    model_kwargs={"temperature": 0.7}
)

# Alternative 2: Local LLama2 Setup
def setup_llama():
    from langchain.embeddings import HuggingFaceInstructEmbeddings
    from langchain.llms import LlamaCpp

    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-xl",
        model_kwargs={"device": "cpu"}
    )

    llm = LlamaCpp(
        model_path="path/to/llama-2-7b.gguf",
        temperature=0.7,
        max_tokens=2000
    )

    return embeddings, llm

def setup_rag_pipeline(embeddings, llm, chunks):
    """
    Set up complete RAG pipeline with calculated k value
    """
    from langchain.vectorstores import FAISS

    # Create vectorstore
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

    # Calculate optimal k
    k_value = calculate_optimal_k(chunks)

    # Setup retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_value})

    # Setup memory
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )

    # Create conversation chain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

    return conversation_chain

def calculate_optimal_k(chunks, total_stocks=621):
    """Calculate optimal k value for retriever"""
    total_chunks = len(chunks)
    avg_chunks_per_stock = total_chunks / total_stocks
    k_value = total_chunks // 2  # Integer division

    print(f"Total chunks: {total_chunks}")
    print(f"Average chunks per stock: {avg_chunks_per_stock:.2f}")
    print(f"Recommended k value: {k_value}")

    return k_value

from langchain.llms import LlamaCpp

llm = LlamaCpp(
    model_path="path/to/llama-2-70b-chat.gguf",
    temperature=0.7,
    max_tokens=2000,
    n_gpu_layers=32  # Adjust based on your GPU
)

from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation",
    model_kwargs={"temperature": 0.7, "max_length": 4096}
)

# # For HuggingFace models
# pip install transformers sentence-transformers accelerate bitsandbytes

# # For LLaMA models
# pip install llama-cpp-python

# Example usage:
def main():
    # Choose your setup:
    embeddings, llm = setup_google_ai()  # or setup_huggingface() or setup_llama()

    # Setup RAG pipeline
    conversation_chain = setup_rag_pipeline(embeddings, llm, chunks)

    # Create chat interface
    def chat(question, history):
        result = conversation_chain.invoke({"question": question})
        return result["answer"]

    return chat

# Requirements for different setups:
REQUIREMENTS = {
    "google": ["google-generativeai", "langchain-google"],
    "huggingface": ["sentence-transformers", "transformers", "huggingface-hub"],
    "llama": ["llama-cpp-python", "transformers"]
}

# Install commands:
"""
# For Google AI:
pip install google-generativeai langchain-google

# For HuggingFace:
pip install sentence-transformers transformers huggingface-hub

# For Llama:
pip install llama-cpp-python transformers
"""