import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import sqlalchemy
import pymysql

DATABASE_LOCATION = "mysql+pymysql://root:root123@localhost/henry_pi"

from utils import get_missings, get_quality

class Config:
    """"""
    datadir="Datasets/"
cf=Config()


def clean_canal_venta(path,delimeter=None):
    df_canalVenta=pd.read_csv(path,delimeter=delimeter)
    df_canalVenta.rename(columns={"CODIGO":"IdCanal","DESCRIPCION":"Canal"},inplace=True)
    df_canalVenta.set_index("IdCanal",inplace=True)

    engine = sqlalchemy.create_engine(DATABASE_LOCATION,echo=True)
    cursor = engine.connect()

    sql_query = """
    create table if not exists canal_venta(
    IdCanal int not null,
    Canal varchar(50) null,
    primary key (IdCanal)
    );
    """
    cursor.execute(sql_query)
    print("Opened database successfully")

    try:
        df_canalVenta.to_sql("canal_venta", engine, index=False, if_exists='append')
    except:
        print("Data already exists in the database")

    cursor.close()
    print("Close database successfully")


def clean_venta(path,delimeter=None):
    df_venta=pd.read_csv(path,delimiter=delimeter)
    df_venta.set_index("IdVenta",inplace=True)
    df_venta["Fecha"]=pd.to_datetime(df_venta["Fecha"],infer_datetime_format=True)
    df_venta["Fecha_Entrega"]=pd.to_datetime(df_venta["Fecha_Entrega"],infer_datetime_format=True)
