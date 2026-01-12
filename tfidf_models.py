import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

df = pd.read_csv("cleaned_reviews.csv")

X = df['review']
y = df['sentiment'].map({'positive':1, 'negative':0})

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LogisticRegression()
lr.fit(X_train, y_train)

nb = MultinomialNB()
nb.fit(X_train, y_train)

print("Logistic Regression accuracy:", accuracy_score(y_test, lr.predict(X_test)))
print("Naive Bayes accuracy:", accuracy_score(y_test, nb.predict(X_test)))
