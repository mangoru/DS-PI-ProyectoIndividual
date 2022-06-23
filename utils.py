from http import server
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


#Calculate missing values percent
def get_missings(df,plot=True,rotation=45,figsize=(10,5),**kwargs):
    labels,values,percent = list(),list(),list()
    if df.isna().sum().sum()>0:
        for column in df.columns:
            if df[column].isna().sum():
                labels.append(column)
                values.append(df[column].isna().sum())
                percent.append(df[column].isna().sum()*100 / df.shape[0])
        #Make a dataframe 
        missings=pd.DataFrame({'Features':labels,'MissingValues':values,'MissingPercent':percent}).sort_values(by='MissingPercent',ascending=False)
        
        if plot:
            plt.figure(figsize=figsize)
            g=sns.barplot(x=missings.Features,y=missings.MissingPercent)
            g.set_title('The Percentage of Missing Values',size=20)
            # for i in g.containers:
            #     g.bar_label(i)
            locs, labels = plt.xticks()
            plt.setp(labels, rotation=rotation,size=16)
            plt.xlabel('Features',size=18)
            plt.ylabel('MissingPercent',size=18)

        return missings
    else:
        return False

def get_correlation_target(X,y,plot=True,rotation=45,vertical=True,figsize=(10,5),**kwargs):
    labels,values = list(),list()

    corr = pd.DataFrame(X.corrwith(y).sort_values(ascending=False))
    #Make a dataframe 
    corr = X.corrwith(y).sort_values(ascending=False)
    corr = pd.DataFrame({"Features":corr.index,"Correlation":corr.values})    
    if plot:
        plt.figure(figsize=figsize)
        if vertical:
            sns.barplot(x=corr["Features"],y=corr["Correlation"],**kwargs).set_title('Features Correlation with Target',size=20)
            locs, labels = plt.xticks()
            plt.setp(labels, rotation=rotation,size=16)
            plt.xlabel('Features',size=18)
            plt.ylabel('Correlation',size=18)
        else:
            sns.barplot(x=corr["Correlation"],y=corr["Features"],**kwargs).set_title('Features Correlation with Target',size=20)
            locs, labels = plt.yticks()
            plt.setp(labels,size=16)
            plt.xlabel('Correlation',size=18)
            plt.ylabel('Features',size=18)

    return corr

def get_correlation_features(df,plot=True,figsize=(10,5),**kwargs):
    corr=df.corr()
    if plot:
        plt.figure(figsize=figsize)
        sns.heatmap(X.corr(),annot=True,fmt=".2",**kwargs)
    return corr


def get_quality(df,columns,plot=True,figsize=(10,5), outliers_only_upper=True):
    df_quality=pd.DataFrame(index=df.index)
    rows=df.shape[0]
    for col in columns:
        df_quality[col+"_quality"]=["Clean" for i in range(rows)] # All rows is initialized as clean
        
        # search outliers
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3-Q1

        upper_bound = (Q3 + (1.5/IQR))
        if outliers_only_upper:
            mask = (df[col] > upper_bound) #& (df[col] < lower_bound)
            df_quality[col+"_quality"][mask]="Outlier" 
            
        else:
            lower_bound = (Q1- (1.5/IQR))
            mask = (df[col] > upper_bound) & (df[col] < lower_bound)
            df_quality[col+"_quality"][mask]="Outlier" 

        # search NaN values
        mask = df[col].isna()
        df_quality[col+"_quality"][mask]="NaN"

        # stats
        df_quality_stats=pd.DataFrame(index=["Clean","Outlier","NaN"])
        for col in df_quality.columns:
            df_quality_stats[col]=df_quality[col].value_counts()
            df_quality_stats[col+"_percent"]=df_quality_stats[col]*100/df_quality.shape[0]
        
    if plot and len(columns)==1:
        g = sns.barplot(x=df_quality_stats.index,y=df_quality_stats[col+"_percent"])
        for i in g.containers:
            g.bar_label(i)
    elif plot and len(columns)>1:
        fig,axs = plt.subplots(1,len(df_quality.columns),figsize=figsize)
        for i,col in enumerate(df_quality.columns):
            g = sns.barplot(x=df_quality_stats.index,y=df_quality_stats[col+"_percent"],ax=axs[i])

            for i in g.containers:
                g.bar_label(i)
            
    return df_quality, df_quality_stats