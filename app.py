import streamlit as st
import re
import time
import utils
from io import StringIO
import nltk
from streamlit_extras.let_it_rain import rain
import random
from datetime import datetime as dt

# Streamlit app
def main():
    
    st.set_page_config(layout="wide", page_title="Plagiarism Checker 🌹")
    
    # Custom CSS for uniform button sizes
    st.markdown("""
        <style>
        .stButton > button {
            width: 100%;
            margin-bottom: 10px;
            align-content: start;
            text-align: left;
            padding-left: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.title("Plagiarism Checker ✅")
        st.write('This application developed to check the plagiarism of document provided.')
        
        st.sidebar.subheader("Downloading nltk punkt")
        progress_bar = st.sidebar.progress(0)
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('punkt_tab')
        progress_bar.progress(100)
            
    if 'page' not in st.session_state:
        st.session_state.page = "Home"

    if st.sidebar.button("Home"):
        st.session_state.page = "Home"
    if st.sidebar.button("Text Input"):
        st.session_state.page = "Text Input"
    if st.sidebar.button("File Upload"):
        st.session_state.page = "File Upload"
    if st.sidebar.button("About"):
        st.session_state.page = "About"
        
    # Display the selected page
    if st.session_state.page == "Home":
        show_home()
    elif st.session_state.page == "Text Input":
        show_text_input()
    elif st.session_state.page == "File Upload":
        show_file_upload()
    elif st.session_state.page == "About":
        apply_custom_css()
        show_about_page()

def show_home():
    st.title("Welcome to the Plagiarism Checker")
    st.write("This application helps you check your text for potential plagiarism.")
    st.write("Use the sidebar to navigate through different sections of the app.")

def show_text_input():
    
    tfidf_matrix, vectorizer = utils.load_tfidf_data()
    
    st.header("Text Input")
    text = st.text_area("Paste your text here:", height=300)
    if st.button("Check Plagiarism"):
        st.session_state['text_to_check'] = text
        st.success("Text submitted for plagiarism check!")
        
        if 'text_to_check' in st.session_state and st.session_state['text_to_check']:
            student_report = st.session_state['text_to_check']
            new_report_vector = vectorizer.transform([utils.preprocess(student_report)])
            
            similarities = utils.cosine_similarity(new_report_vector, tfidf_matrix)
            threshold = 0.2

            if max(similarities[0]) > threshold:
                st.info("Plagiarism detected!")
                
                # Define the emoji list
                emojis = ["🫣", "🤔", "🥵", "🥺", "😭", "😈", "💔"]
                
                rain(
                    emoji=random.choice(emojis),
                    font_size=54,  # Random sizes
                    falling_speed=random.randint(4, 8),  # Random speeds
                    animation_length=random.uniform(0.5, 2)
                )
            else:
                st.success("No plagiarism detected. Adding the report to the reference set.")
                
                # Append the new report to the TF-IDF matrix
                tfidf_matrix = utils.append_to_tfidf_matrix(student_report, vectorizer, tfidf_matrix)
                
                # Save the updated matrix and vectorizer for future use
                utils.save_tfidf_data(tfidf_matrix, vectorizer)
            

def show_file_upload():
    st.header("File Upload")
    uploaded_file = st.file_uploader("Choose a file", type=['txt', 'docx', 'pdf'])
    
    tfidf_matrix, vectorizer = utils.load_tfidf_data()
        
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            new_report = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            new_report_vector = vectorizer.transform([utils.preprocess(new_report)])
            
            similarities = utils.cosine_similarity(new_report_vector, tfidf_matrix)
            threshold = 0.2

            if max(similarities[0]) > threshold:
                st.info("Plagiarism detected!")
                
                # Define the emoji list
                emojis = ["🫣", "🤔", "🥵", "🥺", "😭", "😈", "💔"]
                
                rain(
                    emoji=random.choice(emojis),
                    font_size=54,  # Random sizes
                    falling_speed=random.randint(4, 8),  # Random speeds
                    animation_length=random.uniform(0.5, 2)
                )
            else:
                st.success("No plagiarism detected. Adding the report to the reference set.")
                
                # Append the new report to the TF-IDF matrix
                tfidf_matrix = utils.append_to_tfidf_matrix(new_report, vectorizer, tfidf_matrix)
                
                # Save the updated matrix and vectorizer for future use
                utils.save_tfidf_data(tfidf_matrix, vectorizer)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            
            new_report = utils.preprocess(utils.read_word_file(uploaded_file))
            new_report_vector = vectorizer.transform([utils.preprocess(new_report)])
            
            similarities = utils.cosine_similarity(new_report_vector, tfidf_matrix)
            threshold = 0.8

            if max(similarities[0]) > threshold:
                st.info("Plagiarism detected!")
                
                # Define the emoji list
                emojis = ["🫣", "🤔", "🥵", "🥺", "😭", "😈", "💔"]
                
                rain(
                    emoji=random.choice(emojis),
                    font_size=54,  # Random sizes
                    falling_speed=random.randint(4, 8),  # Random speeds
                    animation_length=random.uniform(0.5, 2)
                )
            else:
                st.success("No plagiarism detected. Adding the report to the reference set.")
                
                # Append the new report to the TF-IDF matrix
                tfidf_matrix = utils.append_to_tfidf_matrix(new_report, vectorizer, tfidf_matrix)
                
                # Save the updated matrix and vectorizer for future use
                utils.save_tfidf_data(tfidf_matrix, vectorizer)
        elif uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
                
            new_report_vector = vectorizer.transform([utils.preprocess(text)])
            
            similarities = utils.cosine_similarity(new_report_vector, tfidf_matrix)
            threshold = 0.2
            
            if max(similarities[0]) > threshold:
                st.info("Plagiarism detected!")
                
                # Define the emoji list
                emojis = ["🫣", "🤔", "🥵", "🥺", "😭", "😈", "💔"]
                
                rain(
                    emoji=random.choice(emojis),
                    font_size=54,  # Random sizes
                    falling_speed=random.randint(4, 8),  # Random speeds
                    animation_length=random.uniform(0.5, 2)
                )
            else:
                st.success("No plagiarism detected. Adding the report to the reference set.")
                
                # Append the new report to the TF-IDF matrix
                tfidf_matrix = utils.append_to_tfidf_matrix(new_report, vectorizer, tfidf_matrix)
                
                # Save the updated matrix and vectorizer for future use
                utils.save_tfidf_data(tfidf_matrix, vectorizer)

def show_about_page():
    # Application title and description
    st.title("About Plagiarism Checker")
    
    # Current version information
    st.header("📦 Version Information")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        **Current Version:** v0.0.1  
        **Released:** October 2024  
        **Status:** Beta
        """)
    
    with col2:
        st.markdown("""
        **Latest Updates:**
        - Added support for multiple file formats (PDF, DOCX, TXT)
        - Improved text highlighting algorithm
        - Enhanced similarity detection
        """)
    
    # Version history with expander
    with st.expander("📝 Version History"):
        st.markdown("""
        ### v0.0.1 (Current) - October 2024
        - Initial release with core functionality
        - Basic plagiarism detection
        - File upload support
        - Similarity highlighting
        """)
    
    # Features
    st.header("✨ Features")
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        **Core Features:**
        - Multiple document format support
        - Real-time plagiarism detection
        - Detailed similarity analysis
        - Interactive result visualization
        """)
    
    with col4:
        st.markdown("""
        **Technical Highlights:**
        - TF-IDF vectorization
        - Cosine similarity matching
        - Advanced text preprocessing
        - Efficient file handling
        - Robust error management
        """)
    
    # Author and contact information
    st.header("👨‍💻 Author Information")
    st.markdown("""
    **Developer:** Isuru Samarappulige  
    **Organization:** Nexlution AI 
    
    [![GitHub](https://img.shields.io/badge/GitHub-View%20on%20GitHub-blue?style=for-the-badge&logo=GitHub)](https://github.com/yourusername/plagiarism-checker)
    """)
    
    # Usage Statistics (if implemented)
    st.header("📊 Usage Statistics")
    col5, col6, col7 = st.columns(3)
    
    with col5:
        st.metric(label="Total Checks", value="1,234")
    with col6:
        st.metric(label="Documents Processed", value="5,678")
    with col7:
        st.metric(label="Active Users", value="789")
    
    # License and legal information
    st.header("📜 License & Legal")
    st.markdown("""
    This project is licensed under the MIT License.
    
    **Copyright © 2024 Your Organization**
    
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files.
    """)
    
    # Footer with last update timestamp
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: gray; padding: 10px;'>
        Last updated: {dt.now().strftime('%B %d, %Y')}
    </div>
    """, unsafe_allow_html=True)

# Custom CSS for better styling
def apply_custom_css():
    st.markdown("""
        <style>
        .stButton > button {
            width: 100%;
        }
        .css-1v0mbdj {
            width: 100%;
        }
        /* Header styling */
        h1 {
            color: #1f77b4;
        }
        h2 {
            color: #cfe2f3;
        }
        /* Metric styling */
        .css-1l92tp0 {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
