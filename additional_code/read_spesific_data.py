import pandas as pd
from .data_clean import read_and_clean_data


def read_conflict_reports():
    dfs = read_and_clean_data()
    conflicts = pd.concat([dfs['battles'], dfs['explotions_or_remote_violence']]).reset_index(drop=True)
    data_from = pd.to_datetime('2020-01-01')
    
    conflicts_israel = conflicts[conflicts['actor1'].str.contains("israel", case=False) | conflicts['actor2'].str.contains("israel", case=False)]
    
    conflicts_israel_palestine= conflicts_israel[((conflicts_israel['country'] =='Israel')|(conflicts_israel['country'] =='Palestine')) & 
                        ((conflicts_israel['admin1'] == 'Gaza Strip') | (conflicts_israel['admin1'] == 'West Bank') | (conflicts_israel['admin1'] == 'HaDarom'))]

    conflicts_israel_palestine['event_date'] = pd.to_datetime(conflicts_israel_palestine['event_date'])
    conflicts_israel_palestine = conflicts_israel_palestine[conflicts_israel_palestine['event_date'] >= data_from]
    conflicts_israel_palestine.reset_index(drop=True, inplace=True)
    conflicts_israel_palestine['year_month'] = conflicts_israel_palestine['event_date'].dt.strftime('%Y-%m')
    
    return conflicts_israel_palestine

def read_annotated_protests_and_riots():
    annotated_protests_and_riots = pd.read_csv("data/riots_and_protests_labled.csv")
    annotated_protests_and_riots = annotated_protests_and_riots[annotated_protests_and_riots['conflict_related'] == 1] 
    data_from = pd.to_datetime('2020-01-01')
    annotated_protests_and_riots['event_date'] = pd.to_datetime(annotated_protests_and_riots['event_date'])
    annotated_protests_and_riots = annotated_protests_and_riots[annotated_protests_and_riots['event_date'] >= data_from]
    annotated_protests_and_riots.reset_index(drop=True, inplace=True)
    annotated_protests_and_riots['year_month'] = annotated_protests_and_riots['event_date'].dt.strftime('%Y-%m')
    return annotated_protests_and_riots

