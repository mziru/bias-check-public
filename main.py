from flask import Flask, render_template, request
from bias_check_utils import (
    generate_response,
    make_retriever,
    make_vector_db,
    id_bias,
    make_table,
    generate_response_fast)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/bias-form')
def form():
    return render_template("bias_form.html")


@app.route('/bias-data', methods=['POST'])
def response_data():
    form_data = request.form
    vectordb = make_vector_db(form_data['URL'])
    qa = make_retriever(vectordb)
    if 'increase_accuracy' in form_data:
        id_results = id_bias(qa)
        response = generate_response(qa, id_results, form_data['URL'])
    else:
        response = generate_response_fast(qa, form_data['URL'])
    table = make_table(response)

    return render_template('bias_data.html', form_data=form_data, response=response, table=table)


if __name__ == '__main__':
    app.run(debug=True)
