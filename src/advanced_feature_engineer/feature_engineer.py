import pandas as pd
import numpy as np
import random
import logging
from collections import Counter
from boruta import BorutaPy
from lightgbm import LGBMRegressor
import gc


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    """
    slightly modified version: of http://stackoverflow.com/a/29546836/2901002

    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees or in radians)

    All (lat, lon) coordinates must have numeric dtypes and be of equal length.

    """
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    a = np.sin((lat2-lat1)/2.0)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2

    return earth_radius * 2 * np.arcsin(np.sqrt(a))


def get_distance(df):

    df['min_latitude'] = min(df['latitude'])
    df['min_longitude'] = min(df['longitude'])
    
    df['max_latitude'] = max(df['latitude'])
    df['max_longitude'] = max(df['longitude'])

    
    for i in ['min','max']:
        for j in ['min','max']:
            
            df['distance_'+i+'_'+j] = df[[i+'_latitude', j+'_longitude', 'latitude','longitude']].apply(lambda x: haversine(x[1], x[0], x[3], x[2]), axis=1)
    
    
    df.drop(columns=['min_latitude','min_longitude','max_latitude','max_longitude'],inplace=True)
    
    
    return df


def price_interpol(grouper,shft):

    grouper = grouper.iloc[:-shft,:]
    grouper['year_month'] = grouper['year_month'].astype(str)+'-01'
    grouper['year_month'] = pd.to_datetime(grouper['year_month'])
    
    months = pd.date_range(
                start=str(grouper['year_month'].min()), end=str(grouper['year_month'].max()), freq='MS')
    
    grouper['year_month'] = grouper['year_month'].astype(str).str[:7]
    
    col = grouper.iloc[:,0]    
    for month in months:
        for group in col.unique():
            if group != group:
                continue
             
            if str(month)[:7] not in grouper[grouper[col.name]==group]['year_month'].values:
                grouper = grouper.append(pd.DataFrame([[group,str(month)[:7],np.nan,np.nan]],
                                                      columns=grouper.columns)).reset_index(drop=True)
                
    grouper = grouper.sort_values(by='year_month').reset_index(drop=True)
    
    grouper['year_month'] = grouper['year_month'].astype(str)+'-01'
    grouper['year_month'] = pd.to_datetime(grouper['year_month'])
    grouper.set_index('year_month',inplace=True)
    
    grouper[grouper.columns[1]] = grouper[grouper.columns[1]].astype(float)
    grouper[grouper.columns[2]] = grouper[grouper.columns[2]].astype(float)
    
    for group in col.unique():
        grouper.loc[grouper[col.name]==group,grouper.columns[1]] = grouper.loc[grouper[col.name]==group,grouper.columns[1]].interpolate(method='time').bfill().ffill()
        grouper.loc[grouper[col.name]==group,grouper.columns[2]] = grouper.loc[grouper[col.name]==group,grouper.columns[2]].interpolate(method='time').bfill().ffill()
    
    grouper = grouper.reset_index()
    
    grouper['year_month'] = grouper['year_month'].astype(str).str[:7] 
    return grouper


def create_group(df,groupers,col,period,shft,interpol=False):
    
    for sh in shft:
            
        temp = df.copy()
        
        temp['price'] = temp.groupby([col,period])['price'].shift(-sh)

        grouper = temp.groupby([col,period]).agg(price_mean=('price','mean'),price_median=('price','median')).reset_index()

        if interpol: grouper = price_interpol(grouper,max(shft))
        
        grouper.rename(columns={'price_mean':col+'_'+str(sh)+'_'+period+'_price_mean','price_median':col+'_'+str(sh)+'_'+period+'_price_median'},inplace=True)
        groupers.append(grouper)
        df = pd.merge(df,grouper,on=[col,period],how='left')
        df.drop_duplicates(subset=['id'],inplace=True)
        
    return df,groupers


def grouping(df):

    groupers = []

    df[df.dtypes[df.dtypes=='object'].index.tolist()] = df[df.dtypes[df.dtypes=='object'].index.tolist()].astype(str)

    for col in ['room_type','neighbourhood','neighbourhood_group']:

        df,groupers = create_group(df,groupers,col,'year',[1])
        df,groupers = create_group(df,groupers,col,'year_month',[3],True)
    
    return df,groupers


def fill_missing(df,groupers, dummies,to_remove=[]):

    df = get_distance(df)

    df[df.dtypes[df.dtypes=='object'].index.tolist()] = df[df.dtypes[df.dtypes=='object'].index.tolist()].astype(str)

    if not to_remove:
        to_remove = ['neighbourhood', 'year','year_month','id','host_id']

    merging_groups = [grouper for grouper in groupers if len([x for x in grouper.columns if 'month' in x])>0]

    df['year'] = df['last_review'].astype(str).str[:4]
    df['year_month'] = df['last_review'].astype(str).str[:7]

    missings = df[df['last_review'].isna()]
    df.dropna(subset=['last_review'],inplace=True)
    
    if len(missings)>0:
        
        idxs = missings.index.tolist()
        for idx in idxs:
            dates = []

            price = missings.loc[idx,'price']

            for grouper in merging_groups:

                col = grouper.columns[1]
                class_ = missings.loc[idx,col]

                prices = grouper[grouper[col] == class_].iloc[:,-2]
                closest_idx = prices.iloc[(prices-price).abs().argsort()[:10]].index
                dates+=grouper.iloc[closest_idx,0].values.tolist()

            date = sorted(Counter(dates).items(), key=lambda item: item[1])[-1][0]

            missings.loc[idx,'year_month'] = str(date)[:7]
            missings.loc[idx,'year'] = str(date)[:4]
        
        df = pd.concat([df,missings],axis=1)
        
        
    for grouper in groupers:

        period = grouper.columns[1]
        col = grouper.columns[0]
        df[period] = df[period].astype(str)
        grouper[period] = grouper[period].astype(str)
        df = pd.merge(df,grouper,on=[col,period],how='left')

    missing_dummies = pd.get_dummies(df[['neighbourhood','neighbourhood_group','room_type']],dummy_na=True)
    df = pd.concat([df,missing_dummies],axis=1)
    df = df.drop(columns=to_remove)
    df[dummies] = 0
    df.drop(columns = missing_dummies.columns[~missing_dummies.columns.isin(dummies)],inplace=True)

    df.drop(columns = 'room_type_nan',inplace=True)
    
    return df


def final_adjustments(df,adj_dummies=[]):    

    df.drop_duplicates(subset=['id'],inplace=True)   
    
    dummies = pd.get_dummies(df[['neighbourhood','neighbourhood_group','room_type']],dummy_na=True)
    df = pd.concat([df,dummies],axis=1)
    df = df.drop(columns=['room_type_nan', 'neighbourhood', 'year','year_month','id','host_id'])
    df.columns = [x.replace('/','_').replace('-','_').replace(',',' ').replace("'",'') for x in df.columns]
    
    if type(adj_dummies)!=list:
        df[adj_dummies.columns[~adj_dummies.columns.isin(df.columns)]] = 0
        df.drop(columns = dummies.columns[~dummies.columns.isin(adj_dummies.columns)],inplace=True)
    
    slicer = (df.dtypes == 'uint8')

    df[df.columns[slicer]] = df[df.columns[slicer]].astype(int)

    return df, dummies.columns.tolist()


def interpolating(df):

    df['year'] = df['last_review'].astype(str).str[:4]
    df['year_month'] = df['last_review'].astype(str).str[:7]

    df = df.sort_values(by=['room_type','neighbourhood','longitude','latitude'])
    df.loc[df['year_month']!='nan','year_month'] = df.loc[df['year_month']!='nan','year_month'].str[:4].astype(float)*1000+(df.loc[df['year_month']!='nan','year_month'].str[5:].astype(int)*99/11).astype(float)
    df.loc[df['year_month']=='nan','year_month'] = np.nan
    df['year_month'] = df['year_month'].astype(float).interpolate(method='linear')
    df['year_month'] = df['year_month'].astype(str).str[:4]+'-0'+(df['year_month'].astype(str).str[5:].astype(float)*11/99).apply(np.ceil).astype(int).astype(str)
    df.dropna(subset=['year_month'],inplace=True)

    df.loc[df['year_month'].str[-3:].isin(['010','011','012']),'year_month'] = df.loc[df['year_month'].str[-3:].isin(['010','011','012']),'year_month'].str.replace('-0','-')
    df.loc[df['year']=='nan','year'] = df.loc[df['year']=='nan','year_month'].str[:4].astype(int)

    return df

def reducing_features(df):

    temp = df.copy()

    temp = temp.drop(columns=['name','neighbourhood_group','room_type','host_name','last_review','reviews_per_month']).dropna()

    y = temp['price']

    X = temp.drop(columns='price')


    model = LGBMRegressor(n_estimators=300, max_depth=50, random_state=42)

    # let's initialize Boruta
    feat_selector = BorutaPy(
        verbose=1,
        estimator=model,
        n_estimators='auto',
        max_iter=50  # number of iterations to perform
    )

    # train Boruta
    # N.B.: X and y must be numpy arrays
    feat_selector.fit(np.array(X), np.array(y))

    # print support and ranking for each feature

    gc.collect()

    cols = list(X.columns[feat_selector.support_])

    logger.info(f'{len(cols)} features remaining')

    return cols