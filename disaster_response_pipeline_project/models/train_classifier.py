
# Pickle &c.
import re
import sys
import pickle
import numpy as np
import pandas as pd

# SQL Alchemy
from sqlalchemy import create_engine

# NLTK Packages
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

# Scikit-Learn
import joblib
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    """
    Load data from sql-database at database_filepath
    
    Parameters:
        database_filepath: string
    Returns:
        X: np-array of text-strings for further processing/tokenization/creating features
        Y: np-array of categories as target for features
        category_names: Categorical name for labeling.
    """
    
    # read data from sql-db into pandas-dataframe
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("DisasterResponse", con=engine)
    X = df.message
    Y = df[df.columns[4:]]
    category_names = Y.columns
    
    return X, Y, category_names

def tokenize(text):
    """
    Input:
        text: Message data for tokenization.
    Output:
        clean_tokens: Result list after tokenization.
    """
    # Clean data using regex
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    """
    Creates a pre-defined sklearn-model/pipeline using
    - CountVectorizes (with function tokenize)
    - TfidTransformer
    - MultiOutputClassifier with
        - RandomForestClassifier with
            - 10 n_estimators
            - 3 min_samples_split
            - gini estimator criterion
            - inverse-document-frequency reweighting enabled
    
    Parameters:
        None
    Returns:
        model: sklearn-Pipeline
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
        
    parameters = {
            'clf__estimator__n_estimators': [50],  # Decision Tree Classifer is also tried here to generate a more accurace model
            'clf__estimator__criterion': ['gini'],
            'clf__estimator__min_samples_split':[2],
            'vect__max_df': [0.25,0.5,0.75],
            'vect__ngram_range':[(1,3),(1,4)],
             }
    
    model = GridSearchCV(pipeline, param_grid=parameters, scoring='precision_samples')
    return model 
    
        
def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate and print (multioutput) sklearn-model 
    using test features and category labels)
    
    Parameters:
        model: (multioutput) sklearn-model
        X_test: np-array of features
        Y_test: np-array of (multioutput) labels
        category_names: list of names of (multioutput) labels
    Returns:
        None
    """
    Y_pred=model.predict(X_test)

    for i in range(33):
        print(Y_test.columns[i], ':')
        print(classification_report(Y_test.iloc[:,i], Y_pred[:,i]))     
        

def save_model(model, model_filepath):
    """
    Saves model as pickle file at model_filepath.
    
    Parameters:
        model: Name of sklearn-model
        model_filepath: Filepath as string
    """
    
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    

def main():
    """
    Main function for training and evaluating model.
    Loads data from sql-database, trains model, evaluates model, saves model.
    
    Parameters:
        None
    Returns:
        None
    """
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