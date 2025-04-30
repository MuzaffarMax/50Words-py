from flask import Flask, request, render_template_string #pip install flask scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
import os

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>TF-IDF Analyzer</title>
</head>
<body>
    <h2>Загрузите текстовый файл</h2>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Загрузить">
    </form>

    {% if table %}
        <h3>Результат:</h3>
        <table border="1">
            <tr>
                <th>Слово</th>
                <th>TF (частота)</th>
                <th>IDF</th>
            </tr>
            {% for row in table %}
            <tr>
                <td>{{ row.word }}</td>
                <td>{{ row.tf }}</td>
                <td>{{ "%.4f" | format(row.idf) }}</td>
            </tr>
            {% endfor %}
        </table>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    table = []
    if request.method == 'POST':
        file = request.files['file']
        if file:
            text = file.read().decode('utf-8')

            # TF-IDF обработка
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform([text])
            words = vectorizer.get_feature_names_out()
            tf = X.toarray()[0]
            idf = vectorizer.idf_

            word_data = []
            for i in range(len(words)):
                word_data.append({
                    'word': words[i],
                    'tf': int(text.lower().split().count(words[i])),
                    'idf': idf[i]
                })

            table = sorted(word_data, key=lambda x: x['idf'], reverse=True)[:50]

    return render_template_string(HTML_TEMPLATE, table=table)

if __name__ == '__main__':
    app.run(debug=True)
