from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        Pclass = int(request.form['Pclass'])
        Sex = 1 if request.form['Sex'] == 'female' else 0
        Age = float(request.form['Age'])
        SibSp = int(request.form['SibSp'])
        Parch = int(request.form['Parch'])
        Fare = float(request.form['Fare'])

        # Prepare features
        features = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare]])
        prediction = model.predict(features)

        result = "Survived 😄" if prediction[0] == 1 else "Did not survive 😢"
        return render_template('result.html', prediction=result)

    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == '__main__':
    # Don't use debug=True in production
    app.run(host='0.0.0.0', port=10000)
