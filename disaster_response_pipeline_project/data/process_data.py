import sys
import sqlite3
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Loads & merges messages & categories datasets from CSV files.
    
    Input:
    messages_filepath - Filepath for CSV file containing messages dataset.
    categories_filepath -  Filepath for CSV file containing categories dataset.
       
    Output:
    df - Pandas DataFrame, containing merged content of message and categories datasets.
    """
    #read datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    '''
    Cleans 'df' Dataframe and returns 'clean_df' 
    
    Parameters:
        df - Pandas DataFrame created in load_data()
    
    Returns:
        clean_df - Pandas DataFrame with cleaned data
    '''
    # Extract message data
    messages_df = df[['id', 'message', 'original', 'genre']]
    
    # extract and clean category column labels 
    categories_df = df['categories'].str.split(";", expand=True)
    categories_colnames = [item[:-2] for item in categories_df.loc[0]]
    categories_df.columns = categories_colnames
    for column in categories_df:
        categories_df[column] = categories_df[column].apply(lambda x: x[-1]).astype(int)
   
        # Replace categories column in df with new category columns.  
    clean_df = pd.concat([messages_df, categories_df], axis=1)
    
    # drop the original related column from `df`
    clean_df.drop(columns=['related'], inplace=True)
        
    # drop duplicates
    clean_df = clean_df.drop_duplicates()

    return clean_df

def save_data(df, database_filename):
    '''
    Save dataframe as SQLite database.
    
    Input:
    df - Cleaned dataframe with merged message and category data.
    database_filename - Text string filename passed as database output name.
       
    outputs:
    None
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, if_exists = 'replace', index=False)  


def main():
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