import sys
import sqlalchemy as sql
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('wordnet') # download for lemmatization
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.externals import joblib

def load_data(database_filepath):
    engine = sql.create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("select * from msg_cat",engine)
    X = df.loc[:,"message"]
    Y = df.iloc[:,4:]
    target_names = list(Y.columns)
    return X.values, Y.values, target_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [25, 50],
        'clf__estimator__criterion': ['entropy', 'gini']
    }

    return GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=-1)
    

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    print("Computing Accuracy for each Category")
    for i in range(36):
        print(category_names[i], " Accuracy: ", accuracy_score(Y_test[:,i],y_pred[:,i]))
    print("\n Classification Report")
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
