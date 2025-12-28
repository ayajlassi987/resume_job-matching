"""Main Streamlit application for Interview Copilot."""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from frontend.api_client import get_client
from frontend.utils import extract_text_from_file

# Page configuration
st.set_page_config(
    page_title="Interview Copilot",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .score-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_client' not in st.session_state:
    st.session_state.api_client = get_client()

# Sidebar navigation
st.sidebar.title("💼 Interview Copilot")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🔍 Resume Matching", "❓ Question Generation", "✅ Answer Evaluation"],
    label_visibility="collapsed"
)

# API Health Check (shown in sidebar)
st.sidebar.markdown("---")
st.sidebar.subheader("API Status")
health_check = st.session_state.api_client.check_health()
if health_check["status"] == "healthy":
    st.sidebar.success("✅ API Connected")
else:
    st.sidebar.error("❌ API Disconnected")
    st.sidebar.caption(health_check.get("message", ""))

# Main content based on selected page
if page == "🏠 Home":
    st.markdown('<div class="main-header">Interview Copilot</div>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Interview Question Generation and Evaluation")
    
    st.markdown("""
    Welcome to **Interview Copilot**, an AI-powered system that helps you:
    - 📋 **Match resumes to job descriptions** using semantic similarity
    - ❓ **Generate tailored interview questions** using RAG (Retrieval-Augmented Generation)
    - ✅ **Evaluate candidate answers** with comprehensive scoring and feedback
    
    ### Quick Start
    
    Use the sidebar to navigate to different features:
    - **Resume Matching**: Match candidate resumes against multiple job descriptions
    - **Question Generation**: Generate personalized interview questions based on resume and job requirements
    - **Answer Evaluation**: Evaluate candidate answers with detailed feedback
    
    ### Features
    
    - 🤖 **Local LLM Support**: Uses open-source LLMs (Ollama, Hugging Face, etc.)
    - 🔍 **Hybrid Search**: Combines semantic and keyword search for better retrieval
    - 🎯 **Reranking**: Cross-encoder reranking for improved question relevance
    - 📊 **Comprehensive Evaluation**: Multiple scoring metrics and detailed feedback
    """)
    
    # API Information
    if health_check["status"] == "healthy":
        st.success("✅ API is running and healthy")
        health_data = health_check.get("data", {})
        if "components" in health_data:
            st.markdown("#### System Components")
            cols = st.columns(len(health_data["components"]))
            for i, (component, status) in enumerate(health_data["components"].items()):
                with cols[i]:
                    if status == "ok":
                        st.success(f"✅ {component.replace('_', ' ').title()}")
                    else:
                        st.error(f"❌ {component.replace('_', ' ').title()}")
    
    # Stats
    if st.button("📊 View System Statistics"):
        stats = st.session_state.api_client.get_stats()
        if stats["status"] == "success":
            stats_data = stats["data"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Questions Generated", stats_data.get("total_questions_generated", 0))
            with col2:
                st.metric("Answers Evaluated", stats_data.get("total_answers_evaluated", 0))
            with col3:
                st.metric("Avg Response Time", f"{stats_data.get('average_response_time', 0):.2f}s")

elif page == "🔍 Resume Matching":
    st.markdown('<div class="main-header">Resume Matching</div>', unsafe_allow_html=True)
    st.markdown("Match candidate resumes against job descriptions using semantic similarity.")
    
    # Input section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Resume")
        resume_option = st.radio("Input method", ["Text", "File Upload"], horizontal=True)
        resume_text = ""
        
        if resume_option == "Text":
            resume_text = st.text_area("Resume text", height=200, placeholder="Paste resume text here...")
        else:
            resume_file = st.file_uploader("Upload resume", type=["txt", "pdf", "docx"])
            if resume_file:
                extracted = extract_text_from_file(resume_file)
                if extracted:
                    resume_text = extracted
                    st.success(f"✅ File uploaded: {resume_file.name}")
                else:
                    st.error("❌ Could not extract text from file")
    
    with col2:
        st.subheader("Job Descriptions")
        job_option = st.radio("Input method", ["Text", "File Upload"], horizontal=True, key="job_option")
        job_descriptions = []
        
        if job_option == "Text":
            num_jobs = st.number_input("Number of job descriptions", min_value=1, max_value=10, value=1)
            for i in range(num_jobs):
                job_text = st.text_area(f"Job Description {i+1}", height=150, key=f"job_{i}")
                if job_text:
                    job_descriptions.append(job_text)
        else:
            job_files = st.file_uploader("Upload job descriptions", type=["txt", "pdf", "docx"], accept_multiple_files=True)
            for job_file in job_files:
                extracted = extract_text_from_file(job_file)
                if extracted:
                    job_descriptions.append(extracted)
                    st.success(f"✅ {job_file.name}")
    
    # Options
    top_k = st.slider("Number of top matches", min_value=1, max_value=10, value=3)
    
    # Match button
    if st.button("🔍 Match Resume", type="primary", use_container_width=True):
        if not resume_text or len(resume_text.strip()) < 10:
            st.error("❌ Please provide a valid resume (at least 10 characters)")
        elif not job_descriptions:
            st.error("❌ Please provide at least one job description")
        else:
            with st.spinner("Matching resume to jobs..."):
                result = st.session_state.api_client.match_resume(resume_text, job_descriptions, top_k)
            
            if result["status"] == "success":
                st.success("✅ Matching completed!")
                matches = result["data"]["matches"]
                
                for i, match in enumerate(matches, 1):
                    with st.expander(f"Match #{i} - Job Index {match['job_index']}", expanded=(i==1)):
                        st.write(f"**Job Description Preview:**")
                        st.write(match.get("job_description", "N/A"))
                        st.caption(f"Score: {match.get('score', 'N/A')}")
            else:
                st.error(f"❌ {result.get('message', 'Unknown error')}")

elif page == "❓ Question Generation":
    st.markdown('<div class="main-header">Question Generation</div>', unsafe_allow_html=True)
    st.markdown("Generate tailored interview questions using RAG (Retrieval-Augmented Generation).")
    
    # Input section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Resume")
        resume_option = st.radio("Input method", ["Text", "File Upload"], horizontal=True, key="q_resume")
        resume_text = ""
        
        if resume_option == "Text":
            resume_text = st.text_area("Resume text", height=200, key="q_resume_text", placeholder="Paste resume text here...")
        else:
            resume_file = st.file_uploader("Upload resume", type=["txt", "pdf", "docx"], key="q_resume_file")
            if resume_file:
                extracted = extract_text_from_file(resume_file)
                if extracted:
                    resume_text = extracted
                    st.success(f"✅ File uploaded: {resume_file.name}")
    
    with col2:
        st.subheader("Job Description")
        job_option = st.radio("Input method", ["Text", "File Upload"], horizontal=True, key="q_job")
        job_text = ""
        
        if job_option == "Text":
            job_text = st.text_area("Job description", height=200, key="q_job_text", placeholder="Paste job description here...")
        else:
            job_file = st.file_uploader("Upload job description", type=["txt", "pdf", "docx"], key="q_job_file")
            if job_file:
                extracted = extract_text_from_file(job_file)
                if extracted:
                    job_text = extracted
                    st.success(f"✅ File uploaded: {job_file.name}")
    
    # Options
    col1, col2, col3 = st.columns(3)
    with col1:
        question_type = st.selectbox(
            "Question Type",
            ["default", "technical", "behavioral", "system_design"],
            format_func=lambda x: x.replace("_", " ").title()
        )
    with col2:
        top_k = st.slider("Retrieved questions (top-k)", min_value=3, max_value=20, value=8)
    with col3:
        st.write("")  # Spacing
        use_hybrid = st.checkbox("Use Hybrid Search", value=True)
        use_reranking = st.checkbox("Use Reranking", value=True)
    
    # Generate button
    if st.button("❓ Generate Questions", type="primary", use_container_width=True):
        if not resume_text or len(resume_text.strip()) < 10:
            st.error("❌ Please provide a valid resume (at least 10 characters)")
        elif not job_text or len(job_text.strip()) < 10:
            st.error("❌ Please provide a valid job description (at least 10 characters)")
        else:
            with st.spinner("Generating questions... This may take a moment."):
                result = st.session_state.api_client.generate_questions(
                    resume_text,
                    job_text,
                    question_type=question_type,
                    top_k=top_k,
                    use_hybrid_search=use_hybrid if use_hybrid else None,
                    use_reranking=use_reranking if use_reranking else None
                )
            
            if result["status"] == "success":
                st.success("✅ Questions generated successfully!")
                data = result["data"]
                questions = data.get("questions", [])
                
                st.subheader("Generated Questions")
                for i, question in enumerate(questions, 1):
                    st.markdown(f"**{i}. {question}**")
                
                # Retrieved questions (expandable)
                if data.get("retrieved_questions"):
                    with st.expander("📚 View Retrieved Questions (used for RAG)"):
                        for i, q in enumerate(data["retrieved_questions"], 1):
                            st.write(f"{i}. {q}")
                
                # Download option
                questions_text = "\n".join([f"{i}. {q}" for i, q in enumerate(questions, 1)])
                st.download_button(
                    "📥 Download Questions",
                    questions_text,
                    file_name=f"interview_questions_{question_type}.txt",
                    mime="text/plain"
                )
            else:
                st.error(f"❌ {result.get('message', 'Unknown error')}")

elif page == "✅ Answer Evaluation":
    st.markdown('<div class="main-header">Answer Evaluation</div>', unsafe_allow_html=True)
    st.markdown("Evaluate candidate answers with comprehensive scoring and feedback.")
    
    # Input section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Interview Question")
        question_text = st.text_area("Question", height=100, placeholder="Enter the interview question...")
        
        st.subheader("Candidate Answer")
        answer_text = st.text_area("Answer", height=200, placeholder="Enter the candidate's answer...")
    
    with col2:
        st.subheader("Evaluation Criteria (Optional)")
        
        expected_keywords_input = st.text_input(
            "Expected Keywords (comma-separated)",
            placeholder="e.g., Python, Flask, REST API"
        )
        expected_keywords = [k.strip() for k in expected_keywords_input.split(",") if k.strip()] if expected_keywords_input else None
        
        expected_skills_input = st.text_input(
            "Expected Skills (comma-separated)",
            placeholder="e.g., Python, Machine Learning"
        )
        expected_skills = [s.strip() for s in expected_skills_input.split(",") if s.strip()] if expected_skills_input else None
        
        context_text = st.text_area(
            "Context (Optional)",
            height=150,
            placeholder="Additional context like resume or job description..."
        )
    
    # Evaluate button
    if st.button("✅ Evaluate Answer", type="primary", use_container_width=True):
        if not question_text or len(question_text.strip()) < 5:
            st.error("❌ Please provide a valid question (at least 5 characters)")
        elif not answer_text or len(answer_text.strip()) < 5:
            st.error("❌ Please provide a valid answer (at least 5 characters)")
        else:
            with st.spinner("Evaluating answer..."):
                result = st.session_state.api_client.evaluate_answer(
                    answer_text,
                    question_text,
                    expected_keywords=expected_keywords,
                    expected_skills=expected_skills,
                    context=context_text if context_text else None
                )
            
            if result["status"] == "success":
                st.success("✅ Evaluation completed!")
                data = result["data"]
                
                # Overall score
                overall_score = data.get("overall_score", 0)
                st.subheader(f"Overall Score: {overall_score * 100:.1f}%")
                st.progress(overall_score)
                
                # Individual scores
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Semantic Similarity", f"{data.get('semantic_similarity', 0) * 100:.1f}%")
                with col2:
                    st.metric("Skill Coverage", f"{data.get('skill_coverage', 0) * 100:.1f}%")
                with col3:
                    st.metric("Keyword Match", f"{data.get('keyword_match', 0) * 100:.1f}%")
                with col4:
                    st.metric("Coherence", f"{data.get('coherence_score', 0) * 100:.1f}%")
                with col5:
                    st.metric("Completeness", f"{data.get('completeness_score', 0) * 100:.1f}%")
                
                # Extracted skills
                if data.get("extracted_skills"):
                    st.subheader("Extracted Skills")
                    st.write(", ".join(data["extracted_skills"]))
                
                # Keywords
                col1, col2 = st.columns(2)
                with col1:
                    if data.get("matched_keywords"):
                        st.success(f"✅ Matched Keywords: {', '.join(data['matched_keywords'])}")
                with col2:
                    if data.get("missing_keywords"):
                        st.warning(f"⚠️ Missing Keywords: {', '.join(data['missing_keywords'])}")
                
                # Feedback
                if data.get("feedback"):
                    with st.expander("💬 Feedback", expanded=True):
                        for feedback_item in data["feedback"]:
                            st.write(f"• {feedback_item}")
                
                # Suggestions
                if data.get("suggestions"):
                    with st.expander("💡 Improvement Suggestions"):
                        for suggestion in data["suggestions"]:
                            st.write(f"• {suggestion}")
            else:
                st.error(f"❌ {result.get('message', 'Unknown error')}")

# Footer
st.markdown("---")
st.caption("Interview Copilot - AI-Powered Interview Question Generation and Evaluation")

