from flask import Flask, render_template, request
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import joblib

app = Flask(__name__)

# Load the pre-trained models and vectorizer
tfidf_vectorizer = joblib.load("tfidf.pkl")
rf_model = joblib.load("best_clf.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def preprocess_text(text):
    text = re.sub('http\S+\s*', ' ', text)  # remove URLs
    text = re.sub('RT|cc', ' ', text)  # remove RT and cc
    text = re.sub('#\S+', '', text)  # remove hashtags
    text = re.sub('@\S+', '  ', text)  # remove mentions
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', text)  # remove punctuations
    text = re.sub(r'[^\x00-\x7f]', r' ', text)  # remove non-ASCII
    text = re.sub('\s+', ' ', text)  # remove extra whitespace
    return text.lower()


def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return ''

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_score', methods=['POST'])
def check_score():
    job_description = request.form.get('job_description', '')
    resume_text = request.form.get('resume_text', '')

    if 'resume_file' in request.files:
        resume_file = request.files['resume_file']
        if resume_file and resume_file.filename.endswith('.pdf'):
            resume_text = extract_text_from_pdf(resume_file)

    job_description = preprocess_text(job_description)
    resume_text = preprocess_text(resume_text)

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform([resume_text, job_description])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    ats_score = round(cosine_sim[0][0] * 100, 2)

    return ({"ATS Score": ats_score})

if __name__ == '__main__':
    app.run(debug=True)






import numpy as np
import pandas as pd
import sklearn

print("numpy", np.__version__)
print("Pandas", pd.__version__)
print("Sklearn", sklearn.__version__)