# After all these stock analysis calls are complete
results_nifty = analyze_stocks(stocks_nifty, output_dir="stock_analysis/reports_v2/nifty_50", max_workers=10, end_date="2025-04-12")
results_midcap = analyze_stocks(stocks_midcap, output_dir="stock_analysis/reports_v2/midcap_150", max_workers=10, end_date="2025-04-12")
results_smallcap = analyze_stocks(stocks_smallcap, output_dir="stock_analysis/reports_v2/smallcap_250", max_workers=10, end_date="2025-04-12")
results_microcap = analyze_stocks(stocks_microcap, output_dir="stock_analysis/reports_v2/microcap_250", max_workers=10, end_date="2025-04-12")

# Add this code to create vectors and store them
print("Creating vector database...")
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings  # Or your preferred embedding model
from langchain.vectorstores import Chroma

# First create the vector creation function
def create_vectors(stock_data, category):
    """Create vectors with enriched metadata"""
    docs = []

    for symbol, data in stock_data.items():
        # Basic metadata
        metadata = {
            "symbol": symbol,
            "company_name": data["company_name"],
            "doc_type": category,  # nifty_50, midcap_150, etc.
            "current_price": str(data["current_price"]),
            "analysis_date": data["actual_end_date"]
        }

        # Add performance metrics to metadata
        if "price_data" in data and "1_year_back" in data["price_data"]:
            metadata["one_year_change"] = str(data["price_data"]["1_year_back"]["price_change_percent"])
        if "price_data" in data and "6_months_back" in data["price_data"]:
            metadata["six_month_change"] = str(data["price_data"]["6_months_back"]["price_change_percent"])

        # Add technical metrics
        if "technical_indicators" in data:
            tech = data["technical_indicators"]
            if "rsi" in tech and tech["rsi"] is not None:
                metadata["rsi"] = str(tech["rsi"])

            # Add momentum rankings
            if "momentum" in tech and "momentum_ratio" in tech["momentum"]:
                mom_data = tech["momentum"]["momentum_ratio"]

                # Add z-scores and rankings
                if "z_scores" in mom_data:
                    if "one_year" in mom_data["z_scores"]:
                        metadata["one_year_z_score"] = str(mom_data["z_scores"]["one_year"])
                    if "six_month" in mom_data["z_scores"]:
                        metadata["six_month_z_score"] = str(mom_data["z_scores"]["six_month"])

                # Add ranks directly to metadata for easier filtering
                if "ranks" in mom_data:
                    if "one_year" in mom_data["ranks"]:
                        metadata["one_year_rank"] = str(mom_data["ranks"]["one_year"])
                    if "six_month" in mom_data["ranks"]:
                        metadata["six_month_rank"] = str(mom_data["ranks"]["six_month"])

                # Add normalized rankings
                if "normalized_z_score" in mom_data and "rank" in mom_data["normalized_z_score"]:
                    metadata["normalized_rank"] = str(mom_data["normalized_z_score"]["rank"])
                    metadata["normalized_score"] = str(mom_data["normalized_z_score"]["score"])

        # Create the text content with detailed information
        content = f"""
        Stock: {data['company_name']} ({symbol})
        Category: {category}
        Current Price: Rs.{data['current_price']:.2f}

        PERFORMANCE METRICS:
        1-Year Change: {data['price_data'].get('1_year_back', {}).get('price_change_percent', 'N/A')}%
        6-Month Change: {data['price_data'].get('6_months_back', {}).get('price_change_percent', 'N/A')}%

        TECHNICAL INDICATORS:
        RSI: {data['technical_indicators'].get('rsi', 'N/A')}
        """

        # Add momentum data if available
        if "technical_indicators" in data and "momentum" in data["technical_indicators"] and "momentum_ratio" in data["technical_indicators"]["momentum"]:
            mom_data = data["technical_indicators"]["momentum"]["momentum_ratio"]
            content += f"""
        MOMENTUM ANALYSIS:
        One Year Momentum Ratio: {mom_data.get('momentum_ratios', {}).get('one_year', 'N/A')}
        Six Month Momentum Ratio: {mom_data.get('momentum_ratios', {}).get('six_month', 'N/A')}

        RANKINGS:
        One Year Rank: {mom_data.get('ranks', {}).get('one_year', 'N/A')}
        Six Month Rank: {mom_data.get('ranks', {}).get('six_month', 'N/A')}
        Normalized Z-Score Rank: {mom_data.get('normalized_z_score', {}).get('rank', 'N/A')}
        """

        # Add strengths and weaknesses
        if "analysis" in data:
            content += "\nSTRENGTHS:\n"
            for strength in data["analysis"].get("strengths", []):
                content += f"- {strength}\n"

            content += "\nWEAKNESSES:\n"
            for weakness in data["analysis"].get("weaknesses", []):
                content += f"- {weakness}\n"

        doc = Document(
            page_content=content,
            metadata=metadata
        )
        docs.append(doc)

    return docs

def chunk_documents(documents):
    """Chunk documents while preserving metadata and context"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True
    )

    chunks = []
    for doc in documents:
        doc_chunks = text_splitter.split_text(doc.page_content)
        for i, chunk_text in enumerate(doc_chunks):
            # Create new metadata with chunk info
            chunk_metadata = doc.metadata.copy()
            chunk_metadata["chunk_id"] = i
            chunk_metadata["total_chunks"] = len(doc_chunks)

            # Add a prefix indicating what the document is about
            if i == 0:
                prefix = f"Stock analysis for {chunk_metadata['company_name']} ({chunk_metadata['symbol']}): "
                chunk_text = prefix + chunk_text

            chunks.append(Document(
                page_content=chunk_text,
                metadata=chunk_metadata
            ))

    return chunks

def create_vector_database(nifty_results, midcap_results, smallcap_results, microcap_results):
    """Create and store vectors for all analyzed stocks"""
    # Process each category
    nifty_docs = create_vectors(nifty_results, "nifty_50")
    midcap_docs = create_vectors(midcap_results, "midcap_150")
    smallcap_docs = create_vectors(smallcap_results, "smallcap_250")
    microcap_docs = create_vectors(microcap_results, "microcap_250")

    # Combine all documents
    all_docs = nifty_docs + midcap_docs + smallcap_docs + microcap_docs

    # Create chunks
    chunks = chunk_documents(all_docs)

    # Create embeddings
    embeddings = OpenAIEmbeddings()  # Or your preferred embedding model

    # Create and persist vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    # Persist to disk
    vectorstore.persist()

    return vectorstore, chunks

# Call the function to create the vector database
vectorstore, chunks = create_vector_database(
    results_nifty,
    results_midcap,
    results_smallcap,
    results_microcap
)

print(f"Vector database created with {len(chunks)} chunks.")

def filter_docs(query):
    """Filter documents based on query content and intent"""
    query = query.lower()

    # Check for specific index mentions
    if "nifty 50" in query or "nifty50" in query:
        return {"doc_type": {"$eq": "nifty_50"}}
    elif "midcap" in query or "mid cap" in query or "mid-cap" in query:
        return {"doc_type": {"$eq": "midcap_150"}}
    elif "smallcap" in query or "small cap" in query or "small-cap" in query:
        return {"doc_type": {"$eq": "smallcap_250"}}
    elif "microcap" in query or "micro cap" in query or "micro-cap" in query:
        return {"doc_type": {"$eq": "microcap_250"}}

    # Check for ranking-related queries - use specialized filters
    if "top ranked" in query or "best performing" in query or "highest ranked" in query:
        if "one year" in query or "1 year" in query or "1-year" in query:
            # Return stocks with best one_year_rank
            return {"one_year_rank": {"$exists": True}}
        elif "six month" in query or "6 month" in query or "6-month" in query:
            # Return stocks with best six_month_rank
            return {"six_month_rank": {"$exists": True}}
        else:
            # Default to normalized rank
            return {"normalized_rank": {"$exists": True}}

    # Extract potential stock symbols from query
    import re
    symbols = re.findall(r'\b[A-Z]{2,10}\b', query)

    # If symbols found but no specific index, search all
    if symbols:
        symbol_filters = []
        for symbol in symbols:
            symbol_filters.append({"symbol": {"$eq": symbol}})

        if len(symbol_filters) == 1:
            return symbol_filters[0]
        else:
            # For multiple symbols, use $or operator
            return {"$or": symbol_filters}

    # Default case - no specific filtering
    return {}

def get_specialized_retriever(query):
    """Create a specialized retriever based on query intent"""
    query_lower = query.lower()

    # Define base parameters
    base_k = min(50, len(chunks) // 20)  # Conservative default

    # Adjust parameters based on query complexity
    if "compare" in query_lower or "versus" in query_lower or " vs " in query_lower:
        # For comparison queries, we need more documents
        k = min(100, len(chunks) // 10)
        fetch_k = k * 2
    elif "top" in query_lower or "best" in query_lower or "highest" in query_lower or "rank" in query_lower:
        # For ranking queries, adjust to get top ranked stocks
        k = 25
        fetch_k = 50

        # Create ranking-specific filters
        if "one year" in query_lower or "1 year" in query_lower or "1-year" in query_lower:
            sort = "one_year_rank"
        elif "six month" in query_lower or "6 month" in query_lower or "6-month" in query_lower:
            sort = "six_month_rank"
        else:
            sort = "normalized_rank"

        # Apply additional category filters
        filter_dict = filter_docs(query)

        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,
                "lambda_mult": 0.7,
                "filter": filter_dict
            }
        )
    else:
        # Default case
        k = base_k
        fetch_k = k * 2

    # Create and return the appropriate retriever
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": 0.7,
            "filter": filter_docs(query)
        }
    )

def query_with_context(query, additional_context=None):
    """Enhance the query with context about what we're looking for"""
    query_lower = query.lower()

    if "rank" in query_lower or "top" in query_lower or "best" in query_lower:
        context = """
        This is a query about stock rankings.
        The ranking data contains:
        - one_year_rank: Rank based on 1-year momentum ratio
        - six_month_rank: Rank based on 6-month momentum ratio
        - normalized_rank: Rank based on normalized z-score

        Lower rank numbers are better (1 is highest rank).
        Please prioritize sources that contain complete ranking information.
        """
        enhanced_query = f"{query}\n\nContext: {context}"

        if additional_context:
            enhanced_query += f"\n\n{additional_context}"

        return enhanced_query

    return query

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI  # Or your preferred LLM

# Create a memory object
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create an LLM instance
llm = ChatOpenAI(temperature=0)

# Create a base retriever for the conversation chain
base_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 15,
        "fetch_k": 30,
        "lambda_mult": 0.7,
        "filter": {}  # Empty filter by default
    }
)

# Create the conversation chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=base_retriever,
    memory=memory,
    verbose=True
)

# Function to process queries
def process_query(user_query):
    # Get enhanced query with context
    enhanced_query = query_with_context(user_query)

    # Get specialized retriever for this query
    specialized_retriever = get_specialized_retriever(user_query)

    # Override the retriever in the chain
    qa_chain.retriever = specialized_retriever

    # Execute the query
    response = qa_chain({"question": enhanced_query})

    return response["answer"]

# Example usage
user_query = "What are the top 5 ranked stocks in Nifty 50 based on normalized Z-score?"
response = process_query(user_query)
print(response)