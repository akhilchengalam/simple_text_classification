import os
import numpy as np
from flask import Flask, request
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from data import target_names, dataset, target

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def hello():
	if request.method == "POST":
		# Builds a dictionary of features and transforms documents to feature vectors and convert our text documents to a
		# matrix of token counts (CountVectorizer)
		count_vect = CountVectorizer()
		X_train_counts = count_vect.fit_transform(dataset)

		# transform a count matrix to a normalized tf-idf representation (tf-idf transformer)
		tfidf_transformer = TfidfTransformer()
		X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

		# training our classifier ; train_data.target will be having numbers assigned for each category in train data
		clf = MultinomialNB().fit(X_train_tfidf, target)
				
		# docs_new = ['I have a Harley Davidson and Yamaha.']
		docs_new = [request.args.get("text")]

		# building up feature vector of our input
		X_new_counts = count_vect.transform(docs_new)
		# We call transform instead of fit_transform because it's already been fit
		X_new_tfidf = tfidf_transformer.transform(X_new_counts)

		# predicting the category of our input text: Will give out number for category
		predicted = clf.predict(X_new_tfidf)

		return {
				"Predicted class": target_names[predicted[0]]
				}
