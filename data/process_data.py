# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Read in the messages and categories files"""
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
#     messages = messages.drop(columns='original')
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages, categories, how='outer', on='id')
    return df



def clean_data(df):
    """Process data so that it is in a suitable format to be used within a ML pipeline"""
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.map(lambda x: x.split('-')[0])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.split('-').map(lambda x: x[1])

        # convert column from string to numeric
        categories[column] = categories[column]

    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    df = df.set_index('id')
    
    # remove duplicates
    df = df.drop_duplicates()
    
    # enforce numerical type for category labels
    col_to_numeric = list(df.columns[~df.columns.isin(['message','genre','original'])])
    df[col_to_numeric] = df[col_to_numeric].apply(pd.to_numeric, downcast='unsigned')
    
    # fixes some boolean values in 'related' that are 2 instead of 1
    df.loc[df['related'] == 2,'related'] = 1

    return df

def save_data(df, database_filename):
    """Create SQL DB and store the processed data in the form of a table"""
    engine = create_engine('sqlite:///' + database_filename)
    try:
        df.to_sql('main_table', engine, index=False, if_exists='replace')  
    except:
        print('Not possible to create Table, it may already exist')
    
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