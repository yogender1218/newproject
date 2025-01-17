from flask import Flask, render_template, request, jsonify
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import joblib

app = Flask(__name__)

# Load the pre-trained models and vectorizer
try:
    tfidf_vectorizer = joblib.load("tfidf.pkl")
    rf_model = joblib.load("best_clf.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except Exception as e:
    print("Error loading models or vectorizer:", e)


def preprocess_text(text):
    """
    Preprocess text by removing URLs, hashtags, mentions, punctuations, non-ASCII characters, and extra whitespace.
    """
    text = re.sub(r'http\S+\s*', ' ', text)  # remove URLs
    text = re.sub(r'RT|cc', ' ', text)  # remove RT and cc
    text = re.sub(r'#\S+', '', text)  # remove hashtags
    text = re.sub(r'@\S+', '  ', text)  # remove mentions
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', text)  # remove punctuations
    text = re.sub(r'[^\x00-\x7f]', r' ', text)  # remove non-ASCII
    text = re.sub(r'\s+', ' ', text)  # remove extra whitespace
    return text.lower()


def extract_text_from_pdf(pdf_file):
    """
    Extract text from a PDF file.
    """
    try:
        reader = PdfReader(pdf_file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ''


@app.route('/')
def index():
    """
    Render the main index page with the form.
    """
    return render_template('index.html')


@app.route('/check_score', methods=['POST'])
def check_score():
    """
    Calculate the ATS score based on job description and resume text.
    """
    job_description = request.form.get('job_description', '')
    resume_text = request.form.get('resume_text', '')

    # If a resume file is uploaded, extract text from it
    if 'resume_file' in request.files:
        resume_file = request.files['resume_file']
        if resume_file and resume_file.filename.endswith('.pdf'):
            resume_text = extract_text_from_pdf(resume_file)

    # Preprocess both texts
    job_description = preprocess_text(job_description)
    resume_text = preprocess_text(resume_text)

    try:
        # Calculate cosine similarity
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform([resume_text, job_description])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        ats_score = round(cosine_sim[0][0] * 100, 2)

        return jsonify({"ATS Score": ats_score})
    except Exception as e:
        print(f"Error calculating ATS Score: {e}")
        return jsonify({"error": "An error occurred while calculating the ATS Score. Please try again."}), 500


if __name__ == '__main__':
    app.run(debug=True)
