# Disaster Response Pipeline Project

This project is part the Udacity datasciente nanodegree. 
Its purpose is to create an API able to classify distress or disaster messages posted on social network according to pre-established categories, by means of a machine learning pipeline using natural language processing.
The API classification is triggered via a web app where the user can enter a message and get classification results. 


### Installation

1. Run the following command to clone the github repository 
    `git clone https://github.com/Djibo-7909/disaster-response-pipeline`

2. You may need to install python additionnal packages using the **requirements.txt** file located at the project's root directory: 
    `pip install -r requirements.txt`

### Instructions:
1. Run the following commands in the project's root directory to generate the database and machine learning model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the **app** directory to run your web app and click on the provided link to launch it.
    `python run.py`

