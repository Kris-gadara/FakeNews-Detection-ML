# @Kriskumar Gadara
# github-Kris-gadara


from flask import Flask, render_template, request
import numpy as np
import re
import nltk
import pickle
import os
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

app = Flask(__name__, template_folder='./templates', static_folder='./static')

# Load model and vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
loaded_model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), 'rb'))
vector = pickle.load(open(os.path.join(BASE_DIR, "vector.pkl"), 'rb'))
lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('english'))


def fake_news_det(news):
    """Detect fake news and return prediction with confidence score."""
    review = str(news)
    review = re.sub(r'[^a-zA-Z\s]', '', review)
    review = review.lower()
    review = nltk.word_tokenize(review)
    corpus = []
    for y in review:
        if y not in stpwrds:
            corpus.append(lemmatizer.lemmatize(y))
    input_data = [' '.join(corpus)]
    vectorized_input_data = vector.transform(input_data)

    # Get prediction
    prediction = loaded_model.predict(vectorized_input_data)

    # Get confidence using decision function
    decision_score = loaded_model.decision_function(vectorized_input_data)
    confidence = abs(float(decision_score[0]))

    # Count non-zero features (how much the model knows about this text)
    num_features = vectorized_input_data.nnz

    return prediction, confidence, num_features


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        message = request.form['news']

        if not message or len(message.strip()) < 10:
            return render_template("prediction.html",
                                   prediction_text="Please enter a longer news article or headline for accurate prediction.")

        pred, confidence, num_features = fake_news_det(message)
        print(f"Prediction: {pred[0]}, Confidence: {confidence:.4f}, Features: {num_features}")

        # Low confidence or very few features = unreliable prediction
        if num_features < 3:
            result = "\u26a0\ufe0f Not enough recognizable content to make a reliable prediction. Please paste a longer news article or headline."
        elif confidence < 0.15:
            if pred[0] == 1:
                result = "\U0001f4f0 Prediction: Possibly Fake News (Low Confidence) - Try pasting the full article for better accuracy."
            else:
                result = "\U0001f4f0 Prediction: Possibly Real News (Low Confidence) - Try pasting the full article for better accuracy."
        else:
            if pred[0] == 1:
                result = "\U0001f6a8 Prediction: This Looks Like Fake News \U0001f4f0"
            else:
                result = "\u2705 Prediction: This Looks Like Real News \U0001f4f0"

        return render_template("prediction.html", prediction_text=result)
    else:
        return render_template('prediction.html', prediction_text="")


if __name__ == '__main__':
    app.run(debug=True)
