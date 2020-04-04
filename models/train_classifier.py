# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
import pickle



def load_data(database_filepath):
    """Load Message data from sql database"""
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('main_table', engine)
    subset = df.columns[~df.columns.isin(['message', 'genre'])]
    df = df.dropna(subset=subset)
                   
    # split into X and Y
    X = df['message']
    Y = df.loc[:,~df.columns.isin(['message', 'original', 'genre'])]
                   
    # convert category label to numeric
    Y = Y.applymap(lambda col: pd.to_numeric(col))
                   
    # extract category names from Y columns
    category_names = list(Y.columns)
                   
    return X, Y, category_names

                   
def tokenize(text):
    """Tokenize function used along CountVectorizer()"""
    text = re.sub(r'[^a-zA-Z0-9]'," ",text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        if tok not in stopwords.words("english"):
            clean_tok = lemmatizer.lemmatize(tok).strip()
            clean_tokens.append(clean_tok)
    return clean_tokens

                   
def build_model():
    """Define ML pipeline. CVGridsearh is also included"""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])
    
    parameters = {
        'vect__max_df':[0.75,1.0],
        'clf__n_estimators':[10,20]
    }

    pipeline = GridSearchCV(pipeline, param_grid=parameters, verbose=10)
    return pipeline

         
                   
def evaluate_model(model, X_test, Y_test, category_names):
    """Print classification report for the model fitted using the input data"""
    Y_pred = model.predict(X_test)
    for i,col in enumerate(Y_test.columns):
        print('CATEGORY: {}\n'.format(col))
        print(classification_report(Y_test[col].values, pd.DataFrame(Y_pred).iloc[:,i]))
                   
def save_model(model, model_filepath):
    """Convert model to pickle object and save it locally"""
    with open(model_filepath,'wb') as file:
        pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        print('This may take a while. A grid search is performed to optimize the model.')
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