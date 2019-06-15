import sys
import pandas as pd 
import sqlalchemy

def load_data(messages_filepath, categories_filepath):
    """
    Load data from csv files
    Merge files on "id" column
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on="id")
    return df, categories

def clean_data(df,categories):
    """
    Create a dataframe of the 36 individual category columns
    Change the column names of the created dataframe
    Convert the category values of the created dataframe into numbers (0 or 1)
    Use the created dataframe to update the original dataframe (1 category column to 36 category columns)
    Drop duplicates
    """
    # create a dataframe of the 36 individual category columns
    categories_columns = categories["categories"]
    categories_ids = categories["id"]
    categories_columns = categories_columns.str.split(pat=';',expand=True)
    # select the first row of the categories dataframe
    column_list_candidates = categories_columns.iloc[0].tolist()
    # use this row to extract a list of new column names for categories.
    column_list = []
    for x in column_list_candidates:
        x = x.split('-')[0]
        column_list.append(x)
    # rename the columns of `categories`
    categories_columns.columns = column_list
    # Convert category values to just numbers 0 or 1.
    for column in categories_columns:
        # set each value to be the last character of the string
        categories_columns[column] = categories_columns[column].str.split('-').str[1]
        # convert column from string to numeric
        categories_columns[column] = pd.to_numeric(categories_columns[column])  
    categories = pd.concat([categories_ids,categories_columns], axis=1)
    # drop the original categories column from `df`
    df.drop("categories", 1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories_columns], axis=1)
    # fill na
    df[categories_columns.columns] = df[categories_columns.columns].fillna(0)
    # drop duplicates
    df["is_duplicate"] = df.duplicated(subset=df.columns.drop("id"))
    df = df[df.is_duplicate != True]
    df = df.drop("is_duplicate",1)
    return df

def save_data(df, database_filename):
    """
    Save dataframe to sql db
    """
    engine = sqlalchemy.create_engine('sqlite:///'+database_filename)
    df.to_sql('df', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df, categories)
        
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