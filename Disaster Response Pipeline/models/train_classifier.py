import sys
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

import pickle
import os
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


class MessageLengthTransformer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.array([len(x) for x in X]).reshape(-1,1)
    

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

    
def load_data(database_filepath):
    path = 'sqlite:///' + database_filepath
    engine = create_engine(path)
    df = pd.read_sql_table(table_name='df', con=engine)
    X = df["message"]
    y = df.loc[:, "related":"direct_report"]
    return X, y, y.columns

    
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

    
def build_model():    
    parameters = {
    'clf__estimator': [
                       #AdaBoostClassifier(n_estimators=50,  learning_rate=0.4),
                       #AdaBoostClassifier(n_estimators=100, learning_rate=0.4),
                       #AdaBoostClassifier(n_estimators=50,  learning_rate=0.8),
                       #AdaBoostClassifier(n_estimators=100, learning_rate=0.8),
                       #AdaBoostClassifier(n_estimators=50,  learning_rate=1),
                       #AdaBoostClassifier(n_estimators=100, learning_rate=1),
                       #RandomForestClassifier(n_estimators=50,  criterion='entropy'),
                       #RandomForestClassifier(n_estimators=100, criterion='entropy'),
                       #RandomForestClassifier(n_estimators=50,  criterion='gini'),
                       #RandomForestClassifier(n_estimators=100, criterion='gini')
                       RandomForestClassifier(n_estimators=10, criterion='gini'),
                       RandomForestClassifier(n_estimators=10, criterion='entropy'),
                       AdaBoostClassifier(n_estimators=10,  learning_rate=1),
                       AdaBoostClassifier(n_estimators=10,  learning_rate=0.5)
        ]
                       
    }

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()) 
            ])),
            ('msg_length', MessageLengthTransformer()),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(estimator=None))
    ])

    cv = GridSearchCV(pipeline, param_grid = parameters)

    return cv


def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)
    for i, category in enumerate(category_names):
        metrics =  classification_report(y_test.iloc[i], y_pred[i])
        print("""category: {}
              {} """.format(category, metrics))

        
def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        gs_model = model.best_estimator_
        
        print('Evaluating model...')
        evaluate_model(gs_model, X_test, y_test, category_names)

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