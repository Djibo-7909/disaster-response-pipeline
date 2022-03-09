# Disaster Response Pipeline Project

This project is part the Udacity datasciente nanodegree. 
Its purpose is to create an API able to classify distress or disaster messages posted on social network according to pre-established categories, by means of a machine learning pipeline using natural language processing.
The API classification is triggered via a web app where the user can enter a message and get classification results. 


## Installation

1. Run the following command to clone the github repository 
    ```
    git clone https://github.com/Djibo-7909/disaster-response-pipeline
    ```

2. You may need to install python additionnal packages using the `requirements.txt` file located at the project's root directory: 
    ```
    pip install -r requirements.txt
    ```

## Instructions:
1. Run the following commands in the project's root directory to generate the database and machine learning model.

    - To run ETL pipeline that cleans data and stores in database
        ```
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        ```
        
    - To run ML pipeline that trains classifier and saves
        ```
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
        ```

2. Run the following command in the **`app`** directory to run your web app and click on the provided link to launch it.
    ```
    python run.py
    ```

## File and directory description
### `data` directory
The **`data`** directory contains the dataset which will help build the machine learning pipeline and the ETL script which deals with this data:
1. `disaster_messsages.csv` :csv file containing a data set of real messages that were sent during disaster events.
2.  `disaster_categories.csv` : csv file containing a classificaion in categories of all messages in the `disaster_messages.csv`. 
3.  `process_data.py` : python ETL script wich loads and merges and cleans messages and categories dataset before storing the output in a SQLite databse.
4. `DisasterResponse.db` : output of the above-mentioned ETL script.

### `models` directory
The **`models`** folder contains the `train_classifier.py` script which: 
1. Loads data from the SQLite database (`DisasterResponse.db`).
2. Splits the dataset into training and test sets.
3. Builds a text processing and machine learning pipeline.
4. Trains and tunes a model using GridSearchCV.
5. Outputs results on the test set.
6. Exports the final model as a pickle file (`classifier.pkl`).