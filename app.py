import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import spacy # <-- BONUS 2: New import

# --- CONFIGURATION ---
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🤖",
    layout="centered",
)

# --- HELPER FUNCTIONS ---

# Function to extract text from different file types
def extract_text_from_file(file):
    text = ""
    if file.type == "application/pdf":
        # Important: Ensure the PDF has selectable text, not just images.
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file.type == "text/plain":
        text = file.read().decode("utf-8")
    return text

# --- BONUS 2: SKILL EXTRACTION SETUP (STEP 2) ---
# Load the spaCy model once
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("SpaCy model not found. In your terminal, please run: 'python -m spacy download en_core_web_sm'")
    nlp = None

# A predefined list of skills (you can expand this list with more skills)
SKILLS_LIST = [
    'python', 'java', 'c++', 'javascript', 'sql', 'nosql', 'react', 'angular', 'vue',
    'machine learning', 'deep learning', 'nlp', 'natural language processing', 'computer vision',
    'data analysis', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'agile', 'scrum', 'api', 'rest'
]

def extract_skills(text):
    """Extracts skills from a text using spaCy and a predefined skill list."""
    if not nlp:
        return set()
    
    doc = nlp(text.lower())
    found_skills = set()
    
    # Check for individual words (tokens) that are skills
    for token in doc:
        if token.text in SKILLS_LIST:
            found_skills.add(token.text)
            
    # Check for multi-word skills (noun chunks)
    for chunk in doc.noun_chunks:
        if chunk.text in SKILLS_LIST:
            found_skills.add(chunk.text)
            
    return found_skills
# --- END OF BONUS 2 SETUP ---


# Function to load data and embeddings (cached for performance)
@st.cache_data
def load_data():
    """Loads job data and pre-computed embeddings."""
    try:
        # Adjust the path if your files are in subfolders
        jobs_df = pd.read_csv('job_title_des.csv/job_title_des.csv')
        job_embeddings = np.load('job_embeddings.npy')
        return jobs_df, job_embeddings
    except FileNotFoundError:
        st.error("Error: Dataset or embeddings not found. Please ensure 'job_title_des.csv' and 'job_embeddings.npy' are in the correct directory.")
        return None, None

# --- MAIN APP LOGIC ---

st.title("🤖 AI-Powered Resume Screener")
st.write("Upload your resume, and we'll find the best job matches for you from our database!")

# Load data and the NLP model
jobs_df, job_embeddings = load_data()

if jobs_df is not None:
    # Use session state to keep the model loaded, avoiding reloads on every interaction
    if 'model' not in st.session_state:
        with st.spinner("Loading NLP model... This may take a moment."):
            st.session_state.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    model = st.session_state.model
    
    # File uploader for the resume
    uploaded_resume = st.file_uploader("Choose a Resume File (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    
    if uploaded_resume is not None:
        # Process the file once it's uploaded
        with st.spinner("Analyzing your resume..."):
            resume_text = extract_text_from_file(uploaded_resume)
            
            if resume_text:
                # Generate embedding for the uploaded resume
                resume_embedding = model.encode([resume_text])
                
                # Calculate cosine similarity against all job descriptions
                cosine_scores = cosine_similarity(resume_embedding, job_embeddings)[0]
                
                # Find the top 5 matches
                top_5_indices = cosine_scores.argsort()[-5:][::-1]
                
                st.success("Analysis complete! Here are your top 5 job matches:")
                
                # --- BONUS 2: NEW DISPLAY LOOP (STEP 3) ---
                # This loop replaces the old one to show matching skills
                for i, index in enumerate(top_5_indices):
                    job_title = jobs_df['Job Title'].iloc[index]
                    job_description = jobs_df['Job Description'].iloc[index]
                    match_score = cosine_scores[index] * 100
                    
                    # Extract skills from both the resume and the job description
                    resume_skills = extract_skills(resume_text)
                    job_skills = extract_skills(job_description)
                    
                    # Find the skills that are in both
                    matching_skills = resume_skills.intersection(job_skills)
                    
                    # Display the match in an expandable box
                    with st.expander(f"**{i+1}. {job_title}** (Match Score: {match_score:.2f}%)"):
                        if matching_skills:
                            st.write("**✅ Matching Skills Found:**")
                            # Display skills in a neat, multi-column layout
                            st.write(", ".join([skill.title() for skill in matching_skills]))
                        else:
                            st.write("**⚠️ No matching skills found from our predefined list.**")
                        
                        st.write("---")
                        st.write("**Job Description Snippet:**")
                        st.write(job_description[:500] + "...")
                # --- END OF BONUS 2 DISPLAY LOOP ---
            else:
                st.error("Could not extract text from the uploaded file. Please try a different file or ensure your PDF contains selectable text.")

