import os
import tempfile
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, BaseTool
import streamlit as st
import PyPDF2
from docx import Document
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
from typing import List, Dict
import re

# Load environment variables
load_dotenv()
os.environ["SERPER_API_KEY"] = "3f056fXXXXXXXXXXXXXXXXXXXXXX"
os.environ["OPENAI_API_KEY"] = "sk-proj-5VIWy1XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
os.environ["OPENAI_MODEL"] = "gpt-4-32k"

# Streamlit App Configuration
st.set_page_config(
    page_title="CrewAI Research Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 CrewAI AI Research Assistant")
st.markdown("Research with internet sources OR upload documents for knowledge-based search.")

# Initialize session state
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'embeddings_model' not in st.session_state:
    st.session_state.embeddings_model = None


# Document Processing Functions
@st.cache_resource
def load_embedding_model():
    """Load sentence transformer model for embeddings"""
    return SentenceTransformer('all-MiniLM-L6-v2')


def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""


def extract_text_from_docx(docx_file):
    """Extract text from Word document"""
    try:
        doc = Document(docx_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading Word document: {str(e)}")
        return ""


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks for better embeddings"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def create_vector_database(documents_data):
    """Create FAISS vector database from documents"""
    if not documents_data:
        return None, []

    # Load embedding model
    model = load_embedding_model()

    all_chunks = []
    metadata = []

    for doc_name, text in documents_data:
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        metadata.extend([{"source": doc_name, "chunk_id": i} for i in range(len(chunks))])

    if not all_chunks:
        return None, []

    # Create embeddings
    embeddings = model.encode(all_chunks)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for similarity

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings.astype(np.float32))

    return index, list(zip(all_chunks, metadata))


def search_documents(query, vector_db, documents, top_k=5):
    """Search documents using vector similarity"""
    if not vector_db or not documents:
        return []

    model = load_embedding_model()

    # Create query embedding
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)

    # Search
    scores, indices = vector_db.search(query_embedding.astype(np.float32), top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(documents):
            chunk, metadata = documents[idx]
            results.append({
                "text": chunk,
                "source": metadata["source"],
                "score": float(score),
                "chunk_id": metadata["chunk_id"]
            })

    return results


# Custom tool that properly inherits from BaseTool
class DocumentSearchTool(BaseTool):
    name: str = "document_search"
    description: str = "Search through uploaded documents to find relevant information"
    vector_db: object = None
    documents: list = []

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, vector_db=None, documents=None, **kwargs):
        super().__init__(**kwargs)
        self.vector_db = vector_db
        self.documents = documents or []

    def _run(self, query: str) -> str:
        """Execute the document search"""
        if not self.vector_db or not self.documents:
            return "No documents available for search."

        results = search_documents(query, self.vector_db, self.documents)

        if not results:
            return "No relevant information found in uploaded documents."

        search_results = "Based on uploaded documents:\n\n"
        for i, result in enumerate(results, 1):
            search_results += f"{i}. From {result['source']} (Score: {result['score']:.3f}):\n"
            search_results += f"{result['text'][:300]}...\n\n"

        return search_results


# Tools
search_tool = SerperDevTool()


# Agents
def create_agents(use_documents=False, doc_search_tool=None):
    if use_documents and doc_search_tool:
        # Document-based researcher
        researcher = Agent(
            role='Document Researcher',
            goal='Extract relevant information from uploaded documents about {topic}',
            verbose=True,
            backstory=(
                "You are an expert at analyzing uploaded documents and extracting relevant information. "
                "You focus on finding specific details from the provided document collection."),
            tools=[doc_search_tool],
            allow_delegation=True
        )

        writer = Agent(
            role='Document-based Writer',
            goal='Create comprehensive reports based on uploaded document analysis about {topic}',
            verbose=True,
            backstory=(
                "You specialize in synthesizing information from document collections into well-structured reports. "
                "You focus on accuracy and cite sources from the uploaded documents."),
            tools=[doc_search_tool],
            allow_delegation=False
        )
    else:
        # Internet-based researcher (original)
        researcher = Agent(
            role='Senior Researcher',
            goal='Uncover groundbreaking technologies in {topic}',
            verbose=True,
            backstory=(
                "Driven by curiosity, you're at the forefront of innovation, eager to explore and share knowledge that could change the world."),
            tools=[search_tool],
            allow_delegation=True
        )

        writer = Agent(
            role='Writer',
            goal='Narrate compelling tech stories about {topic}',
            verbose=True,
            backstory=(
                "With a flair for simplifying complex topics, you craft engaging narratives that captivate and educate, bringing new discoveries to light in an accessible manner."),
            tools=[search_tool],
            allow_delegation=False
        )

    return researcher, writer


# Tasks
def create_tasks(topic, use_documents=False, doc_search_tool=None):
    researcher, writer = create_agents(use_documents, doc_search_tool)

    if use_documents:
        research_task = Task(
            description=(
                f"Analyze the uploaded documents to find relevant information about {topic}. "
                f"Search through the document collection and identify key points, findings, and insights. "
                f"Focus on extracting specific details and evidence from the documents."),
            expected_output="A comprehensive 3 paragraphs long report based on the uploaded documents.",
            agent=researcher,
        )

        write_task = Task(
            description=(
                f"Based on the document analysis about {topic}, compose a detailed report. "
                f"Structure the content in a research paper format with proper citations to the source documents."),
            expected_output="A well-structured report in research paper format including Abstract, Methodology, Literature Review, Results, Conclusion, and Citations based on uploaded documents.",
            agent=writer,
            async_execution=False,
            output_file="document-based-report.md",
        )
    else:
        research_task = Task(
            description=(
                f"Identify the next big trend in {topic}. Focus on identifying pros and cons and the overall narrative. Your final report should clearly articulate the key points, its market opportunities, and potential risks."),
            expected_output="A comprehensive 3 paragraphs long report on the latest AI trends.",
            tools=[search_tool],
            agent=researcher,
        )

        write_task = Task(
            description=(
                f"Compose an insightful article on {topic}. Focus on the latest trends and how it's impacting the industry. This article should be easy to understand, engaging, and positive."),
            expected_output="write the article in Research paper FORMAT where include Abstract,Methodology,literature review which is in table format,Architecture,Results which is in table format, conclusion, citation.",
            tools=[search_tool],
            agent=writer,
            async_execution=False,
            output_file="new-blog-post.md",
        )

    return research_task, write_task


# Crew function
def run_crew_for_topic(topic, use_documents=False, doc_search_tool=None):
    research_task, write_task = create_tasks(topic, use_documents, doc_search_tool)

    researcher, writer = create_agents(use_documents, doc_search_tool)

    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential
    )

    result = crew.kickoff(inputs={'topic': topic})
    return result


# Main Interface
st.sidebar.header("📁 Document Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or Word documents",
    type=['pdf', 'docx', 'doc'],
    accept_multiple_files=True,
    help="Upload multiple PDF or Word documents to create a knowledge base"
)

# Process uploaded files
if uploaded_files:
    with st.sidebar:
        with st.spinner("Processing documents..."):
            documents_data = []

            for uploaded_file in uploaded_files:
                if uploaded_file.type == "application/pdf":
                    text = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                            "application/msword"]:
                    text = extract_text_from_docx(uploaded_file)
                else:
                    continue

                if text.strip():
                    documents_data.append((uploaded_file.name, text))

            if documents_data:
                # Create vector database
                vector_db, processed_docs = create_vector_database(documents_data)
                st.session_state.vector_db = vector_db
                st.session_state.documents = processed_docs

                st.success(f"✅ Processed {len(documents_data)} documents")
                st.info(f"📊 Created {len(processed_docs)} text chunks for search")
            else:
                st.error("❌ No text could be extracted from the uploaded files")

# Research Mode Selection
st.header("🔍 Research Mode")
research_mode = st.radio(
    "Choose your research source:",
    options=["Internet Research", "Document-based Research"],
    help="Select whether to research from internet sources or uploaded documents"
)

# Main Interface
with st.container():
    col1, col2 = st.columns([3, 1])

    with col1:
        topic = st.text_input(
            "Enter a topic for AI Research:",
            placeholder="e.g., AI in Healthcare, Machine Learning in Finance, etc.",
            help="Enter any topic you want to research and generate an article about"
        )

    with col2:
        st.write("")  # Add some spacing
        st.write("")  # Add some spacing
        submit_button = st.button("🚀 Start Research", type="primary")

# Processing and Results
if submit_button and topic:
    if topic.strip():
        use_documents = research_mode == "Document-based Research"

        # Check if documents are available when document mode is selected
        if use_documents and (not st.session_state.vector_db or not st.session_state.documents):
            st.warning("⚠️ Please upload documents first to use document-based research mode.")
        else:
            with st.spinner("🔍 Researching and writing article... This may take a few minutes."):
                try:
                    doc_search_tool = None
                    if use_documents:
                        # Create document search tool using the proper BaseTool inheritance
                        doc_search_tool = DocumentSearchTool(
                            st.session_state.vector_db,
                            st.session_state.documents
                        )

                    # Run the crew
                    result = run_crew_for_topic(topic.strip(), use_documents, doc_search_tool)

                    # Success message
                    mode_text = "document-based" if use_documents else "internet-based"
                    st.success(f"✅ {mode_text.title()} research and article generation completed!")

                    # Display results
                    st.markdown("## 📋 Final Result")

                    # Create tabs for better organization
                    if use_documents:
                        tab1, tab2, tab3 = st.tabs(["📄 Article", "🔍 Document Search", "💾 Download"])
                    else:
                        tab1, tab2 = st.tabs(["📄 Article", "💾 Download"])

                    with tab1:
                        if result:
                            st.markdown(f"### Generated {mode_text.title()} Research Article")
                            st.markdown(result)
                        else:
                            st.warning("No result generated. Please try again.")

                    if use_documents:
                        with tab2:
                            st.markdown("### 🔍 Search Your Documents")
                            search_query = st.text_input("Enter a search query:")

                            if search_query:
                                search_results = search_documents(
                                    search_query,
                                    st.session_state.vector_db,
                                    st.session_state.documents
                                )

                                if search_results:
                                    st.markdown("#### Top 5 Results:")
                                    for i, result in enumerate(search_results, 1):
                                        with st.expander(
                                                f"Result {i} - {result['source']} (Score: {result['score']:.3f})"):
                                            st.markdown(result['text'])
                                else:
                                    st.info("No results found for your query.")

                    download_tab = tab3 if use_documents else tab2
                    with download_tab:
                        if result:
                            # Provide download option
                            file_prefix = "document_based" if use_documents else "internet_based"
                            st.download_button(
                                label="📥 Download Article as Text",
                                data=str(result),
                                file_name=f"{file_prefix}_research_article_{topic.replace(' ', '_').lower()}.txt",
                                mime="text/plain"
                            )

                            # Check if markdown file was created
                            md_file = "document-based-report.md" if use_documents else "new-blog-post.md"
                            if os.path.exists(md_file):
                                with open(md_file, "r", encoding="utf-8") as f:
                                    markdown_content = f.read()

                                st.download_button(
                                    label="📥 Download as Markdown",
                                    data=markdown_content,
                                    file_name=f"{file_prefix}_research_article_{topic.replace(' ', '_').lower()}.md",
                                    mime="text/markdown"
                                )
                        else:
                            st.info("No content available for download.")

                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    st.info("Please check your API keys and try again.")
    else:
        st.warning("⚠️ Please enter a valid topic before submitting.")

elif submit_button and not topic:
    st.warning("⚠️ Please enter a topic for research.")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app now supports two modes:

    🌐 **Internet Research:**
    - Real-time web search
    - Latest trends and technologies
    - Market analysis

    📚 **Document Research:**
    - Upload PDFs and Word documents
    - Create vector embeddings
    - Search through your documents
    - Top 5 relevant results

    **Features:**
    - Multiple file upload support
    - Semantic search with embeddings
    - Professional formatting
    - Download options
    """)

    st.header("🛠️ Configuration")
    st.info("API keys are configured via environment variables")

    if st.session_state.vector_db is not None:
        st.success(f"📊 Vector DB: {len(st.session_state.documents)} chunks ready")

    if st.button("🗑️ Clear Cache & Documents"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.vector_db = None
        st.session_state.documents = []
        st.success("Cache and documents cleared!")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Powered by CrewAI, OpenAI, Streamlit, and SentenceTransformers</div>",
    unsafe_allow_html=True
)