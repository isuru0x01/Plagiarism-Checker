import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from docx import Document
import PyPDF2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np # Import numpy to access vstack
from scipy.sparse import vstack # Import vstack for sparse matrices
# Import necessary libraries
import pickle
from scipy import sparse  # Import the sparse module

nltk.download('punkt')
nltk.download('stopwords')

def preprocess(text):
    # Lowercasing
    # Check if the input is a string before calling lower()
    if isinstance(text, str):
        text = text.lower()
    else:
        # If not a string, potentially handle it (e.g., convert to string)
        # or raise an error
        text = ' '.join(text).lower()  # Joining list elements into a string

    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize text
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    return ' '.join(words)

# Calculation of the TF-IDF vectors from documents
def get_tfidf_vectors(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix

# Check Similarity using new report vector and using reference vectors
def check_similarity(new_report_vector, reference_vectors):
    similarities = cosine_similarity(new_report_vector, reference_vectors)
    return similarities

# Check Jaccard Similarity using two different documents
def jaccard_similarity(doc1, doc2):
    words_doc1 = set(doc1.split())
    words_doc2 = set(doc2.split())
    intersection = words_doc1.intersection(words_doc2)
    union = words_doc1.union(words_doc2)
    return len(intersection) / len(union)

# Read a Word file
def read_word_file(file_path):
    doc = Document(file_path)
    guidelines = [p.text for p in doc.paragraphs if p.text]
    return ' '.join(guidelines) 

# Read a Text file
def read_txt_file(file_path):
    """
    Reads a .txt file and returns the contents as a single string.

    Args:
        file_path (str): Path to the .txt file.

    Returns:
        str: The contents of the file as a single string.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        # Join all lines into a single string, stripping newline characters
        return ' '.join(line.strip() for line in lines if line.strip())


# Reading a PDF document
def read_pdf_file(file_path):
    pdf_file = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    guidelines = ''
    for page_number in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_number]
        guidelines += page.extract_text()
    return ' '.join(guidelines)

# Reading coursework guidelines (assumes it's either a PDF or a Word document)
def read_files(file_path):
    if file_path.endswith('.pdf'):
        return read_pdf_file(file_path)
    elif file_path.endswith('.docx'):
        return read_word_file(file_path)
    else:
        raise ValueError("Unsupported file format. Use PDF or Word files.") 

# Append new report to the existing TF-IDF matrix
def append_to_tfidf_matrix(new_report, vectorizer, tfidf_matrix):
    new_report_vector = vectorizer.transform([preprocess(new_report)])
    tfidf_matrix = vstack([tfidf_matrix, new_report_vector])  # Append new vector
    return tfidf_matrix

# Save the TF-IDF matrix 
def save_tfidf_data(tfidf_matrix, vectorizer):
    sparse.save_npz('tfidf_matrix.npz', tfidf_matrix)
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
# Load the TF-IDF matrix and vectorizer
def load_tfidf_data():
    tfidf_matrix = sparse.load_npz('tfidf_matrix.npz')
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return tfidf_matrix, vectorizer

# Check similarity between a new report and existing reports
def check_similarity(new_report, reference_vectors, vectorizer, threshold=0.8):
    
    new_report_vector = vectorizer.transform([preprocess(new_report)])
    similarities = cosine_similarity(new_report_vector, reference_vectors)
    
    if max(similarities[0]) > threshold:
        return True, max(similarities[0])  # Plagiarism detected
    else:
        return False, max(similarities[0])  # No plagiarism detected
    
# Add the student's report to the tfidf_matrix for future checks
def add_report_to_tfidf(new_report, documents, vectorizer):
    documents.append(preprocess(new_report))
    tfidf_matrix = vectorizer.fit_transform(documents)  # Refit the vectorizer to include the new report
    return tfidf_matrix