import json
import plotly
import plotly.express as px
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data for display
print('load database...');
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterData', engine)
#load score data
df_score = pd.read_sql_table('ScoreTable', 'sqlite:///../data/DisasterResponse.db')
#Modify score dataframe to make it tidy
df_score=pd.melt(df_score, id_vars=['category'], value_vars=['precisions', 'recalls','fscores'],var_name='score', value_name='value')
print('done');

# load model
print('load model...');
model = joblib.load("../models/classifier.pkl")
print('done');

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # Filter dataframe to keep related messages
    # Group by direct_report, request and offer, and count the categories
    df_group=df[(df.related==1)].groupby(['direct_report','request','offer']).sum()
    
    # reset the index of the dataframe
    df_group.reset_index(inplace=True)

    # Filter to keep only the messages classified as "requests",
    # The result is a dataframe with 2 lines, one for direct reports, the other for indirect reports
    # Then, this dataframe is transposed to have columns
    df_group_transpose=df_group[(df_group.request==1)].loc[:, 'aid_related':'other_weather'].transpose()

    # Rename the columns
    df_group_transpose.rename(columns={2: "indirect_report", 5: "direct_report"},inplace=True)
    df_group_transpose.reset_index(inplace=True)

    # create visuals
    graphs = [
        {
            'data': [
                px.treemap(df_group_transpose, path=[px.Constant("indirect_report"), 'index'], values='indirect_report',color='index').data[0]
            ],

            'layout': {
                'title': 'Distribution of categories for indirect requests (related messages only)',
            }
        },
        {
            'data': [
                px.treemap(df_group_transpose, path=[px.Constant("direct_report"), 'index'], values='direct_report',color='index').data[0]
            ],

            'layout': {
                'title': 'Distribution of categories for direct requests (related messages only)',
            }
        },
        {
            'data': [
                px.bar(df_score, x="value", y="category",barmode='group', color='score').data[0],
                px.bar(df_score, x="value", y="category",barmode='group', color='score').data[1],
                px.bar(df_score, x="value", y="category",barmode='group', color='score').data[2]
            ],

            'layout': {
                'title': 'Model score',
                'height': 1200,
                'margin': {'l':150, 'r':0},
                
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