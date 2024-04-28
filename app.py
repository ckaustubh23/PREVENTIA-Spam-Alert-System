from flask import Flask, render_template, request
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
from nltk.stem.porter import PorterStemmer


app = Flask(__name__)

# Loading naive bayes models for emails
tfidf1 = pickle.load(open('C:/Users/KAUSTUBH/Desktop/Preventia/lib/naive_vectorizer.pkl','rb'))
model1 = pickle.load(open('C:/Users/KAUSTUBH/Desktop/Preventia/lib/naive_model.pkl','rb'))

# Loading svm models for sms
tfidf2 = pickle.load(open('C:/Users/KAUSTUBH/Desktop/Preventia/lib/svm_vectorizer.pkl','rb'))
model2 = pickle.load(open('C:/Users/KAUSTUBH/Desktop/Preventia/lib/svm_model.pkl','rb'))

# NAIVE BAYES
def transform_text(text):
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Support Vector Machine
def text_preprocess(text):
    global ps
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)

def stemmer(text):
    text = text.split()
    words = ""
    for i in text:
        stemmer = SnowballStemmer("english")
        words += (stemmer.stem(i))+" "
    return words

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_sms = request.form['content']
    option = request.form['option']

    if option == 'EMAIL':
        transformed_sms1 = transform_text(input_sms)
        vector_input1 = tfidf1.transform([transformed_sms1])
        result1 = model1.predict(vector_input1)[0]
        if result1 == 1:
            result_text = "Email Content is Spam"
        else:
            result_text = "Email Content is Not Spam"
    elif option == 'SMS':
        transformed_sms2 = text_preprocess(input_sms)
        transform_sms = stemmer(transformed_sms2)
        vector_input2 = tfidf2.transform([transform_sms])
        result2 = model2.predict(vector_input2)[0]
        if result2 == 1:
            result_text = "SMS Content is Spam"
        else:
            result_text = "SMS Content is Not Spam"

    return render_template('result.html', result=result_text)

if __name__ == '__main__':
    app.run(debug=True)
