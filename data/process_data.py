import sys
import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load data from input csv files into a panda dataframe.

    Args:
    messages_filepath: path to csv file containing all disaster messages.
    categories_filepath: path to csv file containing classification of disaster messages into categories.

    Returns:
    Dataframe containing the merge of messages and classification.
    """

    # load messages and categories dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories)
    
    return df

def clean_data(df):
    """Clean the input dataframe of disaster messages and categories

    Args:
    df: panda dataframe containing messages and categories.

    Returns:
    Dataframe of messages merged with classification into categories.
    Each category is in a specific column containing either 0 or 1.
    The  dataframe is cleaned from duplicates and messages without any category.
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';',expand=True)
    
    # select the first row of the categories dataframe
    row = categories.head(1)
    
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x.str.split(pat='-').str[0],axis=0).transpose()
    
    # rename the columns of `categories`
    categories.columns = category_colnames[0]
    
    #Convert category values to numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split("-").str[1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    df.drop(columns=['categories'],inplace=True);    
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)    
        
     # drop duplicates
    df.drop_duplicates(inplace=True)

    #drop lines with related column = 2 (lines with no classification)
    df =df[df.related!=2]
    
    return df
    
def save_data(df, database_filename):
    """Save the panda dataframe into a sqlite database

    Args:
    df: panda dataframe containing cleaned messages and categories.
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterData', engine, index=False, if_exists='replace')  


def main():
    """
    ETL of disaster message and associated categories:
    1. Load csv data into panda dataframe.
    2. Transform and clean dataframe.
    3. Save dataframe into sqlite database.
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()