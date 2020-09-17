from flask import Flask, render_template, request
import numpy as np
import pickle

app=Flask(__name__,template_folder='templates')

model = pickle.load(open('model.pkl', 'rb'))
s = pickle.load(open('standardscaler.pkl', 'rb'))
map_dict = {
    0:"Iris-setosa",
    1:'Iris-versicolor',
    2:'Iris-virginica'
}

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    arr = [float(x) for x in request.form.values()]
    arr = [np.array(arr)]
    arr = s.transform(arr)
    pred = model.predict(arr)
    return render_template('index.html', prediction_text = f'This is a {map_dict[pred[0]]} type of flower')

if __name__ == '__main__':
    app.run(debug=True)

