from flask import Flask,render_template, jsonify, request, redirect, json, url_for , flash
from ML_BookRecommender import recommend_book
import pandas as pd
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search',methods= ['POST'])
def searchBooks():
    upperCost = request.form['cost']
    lowerRating = request.form['stars']
    genre = request.form['genre']
    keywords = request.form['keywords']

    recommended_books = recommend_book(upperCost,lowerRating,genre,keywords)
    
    book_list = []
    for df in recommended_books:
        if not df.empty:
            for _, row in df.iterrows():
                # Drop rows with NaT values
                row = row.dropna()

                # Convert remaining row to dictionary
                book_dict = row.to_dict()
                book_list.append(book_dict)
    content = jsonify(book_list);
    
    return content


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

