import sys
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])


# import libraries
import pandas as pd
import numpy as np
import re
import pickle
from sqlalchemy import create_engine
from sklearn.datasets import make_multilabel_classification
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

nltk.download('averaged_perceptron_tagger')

def load_data(database_filepath):
    """Load data from the sqlite database
    Args:
    database_filepath: path to sqlite database containing cleaned messages and categories. 

    Returns:
    X: messages.
    Y: classification in categories.
    category_names: list of all caterogies.
    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterData', 'sqlite:///'+database_filepath) 
    X = df.message.values
    Y = df.loc[:, 'related':'direct_report'].values
    category_names=df.columns[4:].tolist()
    return X,Y,category_names

def tokenize(text):
    """Function which tokenizes a given text:

    Args:
    text: disaster message to be tokenized. 

    Returns:
    clean_tokens_filtered: list of words filtered from message tokenization.
    """

    stop_words = stopwords.words("english")# Define stop words 
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())# Normalize text   
    tokens = word_tokenize(text) # Tokenize
    lemmatizer = WordNetLemmatizer()
    
    # lemmatize and remove stop words
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words] # lemmatize words and remove stop words
    clean_tokens_filtered = [wt for (wt, tag) in pos_tag(clean_tokens) if tag in ['VB','VBP','VBG','VBZ','NN','NNS']] # keep verbs and nouns
    return clean_tokens_filtered

def build_model(X_train, Y_train):
    """Machine learning pipeline:

    Returns:
    pipeline: list of words filtered from message tokenization.
    """

    #build machine learning pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('classifier' , MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1)),
    ])
    
    #Use gridsearchCV to find best parameters for the model
    #NB: range of parameters reduced to limit the computation time and limit the size of the pickle file.
    print(' opimizing ml model (gridsearchCV)...');
    parameters ={
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.75, 1.0),
        'vect__max_features': (None, 5000),
        'tfidf__use_idf': (True, False),
        'classifier__estimator__n_estimators': [5,10],
        'classifier__estimator__min_samples_split': [2, 3],
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    cv.fit(X_train , Y_train)
    # print best parameters
    print(' output of GridSearchCV gives parameters: '+ str(cv.best_params_))  
    #apply best parameters to pipeline.
    pipeline.set_params(**cv.best_params_)

    return pipeline

def evaluate_model(model, X_test, Y_test, category_names,database_filepath):
    """ Evaluate machine learning model and store precision scores in the disaster messages database.

    Args:
    model: machine learning model.
    X_test: subset of messages for test.
    Y_test: subset of classification of messages in X_test.
    category_names: list of categories.
    database_filepath: path to disaster messages database.

    """
    #predict model
    Y_pred= model.predict(X_test)

    #print scores
    print('Printing classification report for each category...')
    for i in range(Y_test.shape[1]):
        print('Category: '+category_names[i])
        print(classification_report(Y_test[:, i],Y_pred[:, i]))

    #save scores in arrays
    precisions = np.array([])
    recalls = np.array([])
    fscores = np.array([])
    for i in range(Y_test.shape[1]):
        precision,recall,fscore,support=score(Y_test[:, i],Y_pred[:, i],average='weighted')
        precisions=np.append(precisions,precision)
        recalls=np.append(recalls,recall)
        fscores=np.append(fscores,fscore)    
    #create dataframe of scores and save is as a table in the DisasterResponse database
    df_score = pd.DataFrame({'category':np.array(category_names),'precisions':precisions,'recalls':recalls,'fscores':fscores})
    engine = create_engine('sqlite:///'+database_filepath)
    df_score.to_sql('ScoreTable', engine, index=False, if_exists='replace')

def save_model(model, model_filepath):
    """ Save machine learning model into pickle file.

    Args:
    model: machine learning model.
    model_filepath: path to pickle file.

    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    """ Machine learning pipeline:
    1. Load data from disaster message database
    2. extract data from database and create training and test subsets
    3. Build and train machine learning model
    4. Evaluate model and store precision scores in the disaster message database
    5. Store the model into a pickle file.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train, Y_train)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names,database_filepath)

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