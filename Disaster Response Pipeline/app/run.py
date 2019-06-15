import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)

def tokenize(text):
    """
    Tokenization function
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class MessageLengthTransformer(BaseEstimator, TransformerMixin):
    """
    In this class we create a transformer that calculates the Message Length
    for each message
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.array([len(x) for x in X]).reshape(-1,1)
    

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    In this class we create a starting verb extractor
    """
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
    """
    This function is used to load data
    """
    path = 'sqlite:///' + database_filepath
    engine = create_engine(path)
    df = pd.read_sql_table(table_name='df', con=engine)
    X = df["message"]
    y = df.loc[:, "related":"direct_report"]
    return X, y, y.columns


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('df', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    message_lengths = df['message'].apply(len).value_counts()
    
    new_df = pd.DataFrame(df['message'].apply(tokenize).tolist()).stack()
    new_df = new_df.reset_index()
    new_df.drop(columns=["level_0", "level_1"], inplace=True)
    new_df.columns = ['words']
    word_counts = new_df['words'].value_counts().head(35)

    graphs = [  
   {  
      "data":[  
         {  
            "x":genre_names,
            "y":genre_counts,
            "type":"bar",
            "marker":{"color": "rgb(224, 102, 102)"}
         }
      ],
      "layout":{  
         "title":"Distribution of Message Genres",
         "yaxis":{  
            "title":"Count"
         },
         "xaxis":{  
            "title":"Genre"
         }
      }
   },
   {  
      "data":[  
         {  
            "x":message_lengths.index,
            "y":message_lengths.values,
            "type":"bar",
         }
      ],
      "layout":{  
         "title":"Message Length Frequency Graph",
         "yaxis":{  
            "title":"Message Length(Characters)"
         },
         "xaxis":{  
            "title":"Frequency",
            "range": [
                0,
                500
            ]
         }
      }
   },
   {  
      "data":[  
         {  
            "x":word_counts.index,
            "y":word_counts.values,
            "type":"bar",
            "marker":{"color": "rgb(87, 77, 77)"}
         }
      ],
      "layout":{  
         "title":"Message Common Words Frequency Graph",
         "xaxis":{  
            "title":"Words"
         },
         "yaxis":{  
            "title":"Word Frequency"
         }
      }
   }
]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
