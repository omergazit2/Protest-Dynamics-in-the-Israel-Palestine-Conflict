import re
import pandas as pd


def remove_year_parentheses_from_actors(df, drop_undifined= True, drop_nulls=True):
    """clean actors columns by removing year parentheses and droping rows with null values

    Args:
        df (pd.DataFrame): dataframe from ACLED data.
        drop_undifined (bool, optional): drop undifined groups. Defaults to True.
        drop_nulls (bool, optional): drop actors that are not spesified. Defaults to True.
    
    Returns:
        df (pd.DataFram): cleaned dataframe.
    """
    
    def remove_year_parentheses(text):
        if pd.isnull(text):
            return pd.NA
        if drop_undifined and 'Unidentified' in text:
            return pd.NA
        pattern = r'(\s*\(\d{4}[-–—](?:\d{4}|)\)|\s*\([-–—]\d{4}\))(.*)'
        result = re.sub(pattern, '', text).strip()
        return result


    df['actor1'] = df['actor1'].apply(remove_year_parentheses)
    df['actor2'] = df['actor2'].apply(remove_year_parentheses)
    
    if drop_nulls:
        df = df.dropna(subset=['actor1', 'actor2'])

    return df

def drop_unnecessary_columns(df):
    """drop unnecessary columns from ACLED data
    """
    df = df.drop(columns=['timestamp', 'tags', 'geo_precision', 'admin3', 'admin2', 'location', 'time_precision'])
    return df


def read_and_clean_data():
    """Read ACLED data after clean
    
    Returns:
        dict: dictionary of cleaned dataframes [battles, explotions_or_remote_violence, riots, protests, violence_against_civilians, riots_US (annotated), protests_US(annotated)]
    """
    dfs = {
    "battles" : pd.read_csv('data/battles.csv'),
    "explotions_or_remote_violence" : pd.read_csv('data/explosions_or_remote_violence.csv'),
    "riots_and_protests" : pd.read_csv('data/riots_and_protests.csv'),
    "violence_against_civilians" : pd.read_csv('data/violence_against_civilians.csv')
    }
    
    # saperate_riots_and_protests
    riots_and_protests = dfs['riots_and_protests']
    protests = riots_and_protests[riots_and_protests['event_type'] == 'Protests']
    riots = riots_and_protests[riots_and_protests['event_type'] == 'Riots']
    
    dfs['riots'] = riots
    dfs['protests'] = protests
    dfs.pop('riots_and_protests')
    
    # drop unnecessary columns
    for k, v in dfs.items():
        dfs[k] = drop_unnecessary_columns(v)
        
    # remove Identity militias from actors
    for k, v in dfs.items():
        dfs[k] = dfs[k][~dfs[k]['interaction'].str.contains('identity militia', case=False)]
    
    # remove year parentheses from actors
    remove_year_parentheses_from_actors_nalls_from = ['battles', 'explotions_or_remote_violence', 'violence_against_civilians']
    for k, v in dfs.items():
        if k in remove_year_parentheses_from_actors_nalls_from:
            dfs[k] = remove_year_parentheses_from_actors(v).reset_index(drop=True)
        else:
            dfs[k] = remove_year_parentheses_from_actors(v, drop_nulls = False, drop_undifined=False)
    
    # save cleaned data
    if __name__ == '__main__':
        for k, v in dfs.items():
            v.to_csv(f'data_clean/{k}.csv', index=False)
    
    dfs['riots_US'] = pd.read_csv('data/riots_US_ranked.csv')
    dfs['protests_US'] = pd.read_csv('data/protests_US_ranked.csv')
     
    return dfs


    
    
    