# Importazione basic modules.

import sys
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sn
import datetime
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
from sqlalchemy.types import String, Date, DateTime, Float
import bcrypt
import json
import traceback
import getpass

# per riadattare la fuzione wrapper in modo da aggiornare la docstring.
from functools import wraps

# Per produrre grafici visivamente più piacevoli
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Per ignorare "warnings" poco utili
import warnings
from IPython.core.display import display
warnings.filterwarnings(action="ignore", message="^internal gelsd")

# Impostare un seed per produrre risultati riproducibili
np.random.seed(42)

url_regioni = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv'


def check_password_decorator(func):
        '''
        Decoratore per effettuare il check della password.
        '''
        __hashedPassword = b'$2b$12$7BFNKqN9XC5hLTPCL0sXiuJHqraad7Frg.1nfWgGlIekVLnwJPwcO'
        @wraps(func)
        def func_wrapper(*args, **kargs):
            try:
                if bcrypt.checkpw(kargs['password'].encode(), __hashedPassword):
                    return func(*args, **kargs)
                else:
                    print('ERRORE PASSWORD ERRATA!')
            except KeyError:
                print('Please, insert the PASSWORD attribute!')
                
        return func_wrapper


class CovidDataHandler:
    '''
    Classe generale per la gestione delle funzionalità del progetto.
    ''' 
    def __init__(self, db):
        '''
        Costruttore che inizializza il DB e alcuni attributi di istanza. 
        Necessita come input l'istanza della classe "DataBase".
        '''
        self.db = db
        self.csv_file_name = ''
        self.analysis_file_name = ''

    def retrieve_data(self):
        '''
        Punto 1: Metodo per effettuare il download dei dati dal repository github 
        della protezione civile italiana riguardo l outbreak di SARS-Cov-2. 
        Il metodo effettua anche un blando preprocessing per ordinare le righe in 
        base alla regione e alla data ed in più genera una nuova feature nel dataset
        che è il ratio tra il totale dei casi e i tamponi effettuati alla data x.     
        '''
        try:
            print('Trying to retrieve data from {}'.format(url_regioni))
            self.df_regioni = pd.read_csv(url_regioni)
            print('Data retrieved successfully from %s'%(url_regioni))
            print('Doing some elaboration on data....')
            self.df_regioni["data"] = pd.to_datetime(self.df_regioni["data"])
            # per maggiore leggibilità ordiniamo per regione e per data.
            self.df_regioni_sorted = self.df_regioni.sort_values(by=['denominazione_regione', 'data'])
            # creazione nuove caratteristiche: ratio tra totale dei casi e tamponi effettuati.
            self.df_regioni_sorted['totale_casi-tamponi_ratio'] = self.df_regioni_sorted['totale_casi'] / self.df_regioni_sorted['tamponi']
            self.df_regioni_sorted.set_index("data", inplace=True)
            print('Data retrieved and successfully elaborated! Data available in "df_regioni_sorted" attribute.')            
        except Exception:
            print('It hasn\'t been possible to retrieve the data!')
            traceback.print_exc()

    @check_password_decorator
    def save_retrieved_to_csv(self, file_name, password):
        '''
        Punto 1a. Metodo per salvare i dati ottenuti nel Punto1 in un file csv.
        Necessita l'inserimento del nome/path del file con il quale si desidera salvare
        i dati ottenuti dal repository e la password.
        '''
        self.csv_file_name = file_name
        self.df_regioni_sorted.to_csv(self.csv_file_name)
        print('File "{}" saved successfully!'.format(self.csv_file_name))

    @check_password_decorator
    def load_retrieved_from_csv(self, password):
        '''
        Punto 1b. Metodo per caricare i file scaricati in precedenza e salvati in un file locale csv.
        Necessita dell'inserimento della password e che sia stato già creato il file csv locale dei dati
        scaricati in precedenza con il metodo "save_retrieved_to_csv" e che la configurazione a questi sia stata
        salvata nel file "raw_data.config" tramite il metodo "save_configuration_raw_data".
        '''
        with open('raw_data.config', 'r') as f:
            self.csv_file_name = f.readline()
        self.df_regioni_sorted = pd.read_csv(self.csv_file_name)
        self.df_regioni_sorted["data"] = pd.to_datetime(self.df_regioni_sorted["data"])
        self.df_regioni_sorted.set_index("data", inplace=True)     
        print('File "{}" loaded successfully! It is available in the attribute df_regioni_sorted'.format(self.csv_file_name))
        return self.df_regioni_sorted

    @check_password_decorator
    def save_configuration_analysis(self, password):
        '''
        Punto 2. Metodo per salvare su un file analysis.config la configurazione per l'accesso al DB MongoDB
        che si userà per salvare le analisi come documenti JSON. Utilizza una funzionalità della classe DataBase.
        Necessita dell'inserimento della password.
        '''
        self.db.save_db_config()

    @check_password_decorator
    def save_configuration_raw_data(self, password):
        '''
        Punto 2bis. Metodo per salvare su un file raw_data.config la configurazione per caricare il file csv 
        con i dati scaricati e salvati. (punto aggiunto da me per non usare semplicemente un parametro passato
        al metodo per caricare il file csv). Necessita dell'inserimento della password.
        '''
        if self.csv_file_name != '':
            with open("raw_data.config", 'w') as f:
                f.write(self.csv_file_name)
                print("Configuration's file for raw data retrieved saved correctly.")
        else:
            print("Error! Data haven't yet been saved in a valid file. Save the data in a valid url using the method 'save_retrieved_to_csv' and then try again to save the configuration.")

    @check_password_decorator       
    def selection_data_subset(self, password, start_date, end_date, lista_regioni, feature_da_estrarre):
        '''
        Punto 3. Metodo per selezionare un sottoinsieme di dati a partire dal dataset iniziale.
        Effettua una selezione sulla base di un intervallo di date, di una o più regioni fornite come lista e
        una feature da usare per l analisi. Il DataFrame risultante sarà salvato come variabile di istanza.
        Necessita di inserire:
        start_date, end_date, lista_regioni e la feature_da_estrarre (es. tamponi, ecc.ecc.).
        Necessita dell'inserimento della password.
        '''
        self.__start_date = start_date
        self.__end_date = end_date
        self.subset = self.df_regioni_sorted[start_date: end_date].query("denominazione_regione == {}".format(lista_regioni))[["denominazione_regione"]+[feature_da_estrarre]]
        return self.subset

    @check_password_decorator       
    def analize_data_subset(self, password):
        '''
        Punto 4. Metodo per analizzare il sottoinsieme di dati selezionato con il metodo del 
        punto 3. Utilizza il DataFrame generato con il metodo al punto 3 per generare un DataFrame con 
        i risultati dell'analisi di aggregazione raggruppando i dati per regioni (se queste sono più di una).
        Necessita dell'inserimento della password.
        '''
        self.subset_aggr = self.subset.groupby("denominazione_regione").agg(["mean", "median", "skew", "max", "min"])
        # rinominazione delle colonne per il successivo salvataggio nel DB.
        self.subset_aggr[(self.subset_aggr.columns[0][0], "start_date_selected")] = self.__start_date
        self.subset_aggr[(self.subset_aggr.columns[0][0], "end_date_selected")] = self.__end_date
        self.subset_aggr[(self.subset_aggr.columns[0][0], "analysis_date")] = pd.to_datetime([datetime.datetime.now() for i in range(self.subset_aggr.shape[0])])
        self.subset_aggr[(self.subset_aggr.columns[0][0], "feature_name")] = [self.subset_aggr.columns[0][0] for i in range(self.subset_aggr.shape[0])]
        self.subset_aggr.columns = self.subset_aggr.columns.get_level_values(1)
        return self.subset_aggr

    @check_password_decorator       
    def save_analisys_to_db(self, password):
        '''
        Punto 5b. Metodo per salvare in un database il risultato dell'analisi al punto 4 usando la
        configurazione del file "analsysis.config" generato con il metodo al punto 2 "save_configuration_analysis".
        Ogni analisi sarà un nuova riga della tabella analisi nel DB in aggiunta al timestamp del giorno in cui 
        è stata effettuata l'analisi di aggregazione. Necessita dell'inserimento della password.
        '''
        try:
            self.db.save_analysis(self.subset_aggr)
            print("Analysis correctly saved into the DB table.")
        except Exception:
            print("Errore! Salvataggio nel DB non riuscito!")
            traceback.print_exc()

    @check_password_decorator       
    def produce_graphs(self, password):
        '''
        Punto 6. Metodo per produrre i grafici relativi al sottoinsieme di dati selezionati 
        sullo stesso sottoinsieme di dati. Necessita dell'inserimento della password.
        '''
        self.subset.groupby("denominazione_regione")[self.subset.columns[1]].plot(legend=True, title= self.subset.columns[1], 
                                                                                    figsize=(16,5), logy=True)



class Database:
    '''
    Classe per la gestione del database Postgresql tramite SQLAlchemy.
    '''
    def __init__(self):
        '''Legge della configurazione per la connessione al DB (se presente) e 
        inizializza la il pool di connessioni al DB. Se il DB richiesto non è presente
        nel cluster inserito come input, lo stesso verrà creato in automatico.
        Se il file di configurazione non è presente chiede in input USERNAME, PASSWORD e 
        HOST (nella forma ad es: 127.0.0.1:xxxx) e si connetterà al DB utilizzando questi dati 
        appena inseriti. 
        Bisognerà quindi salvare la configurazione tramite il metodo della classe 
        "CovidDataHandler.save_configuration_analysis()"
        '''
        try:
            with open('db_config.json', 'r') as f:
                self.db_config = json.loads(f.readline())
            self.db_string = "postgresql+psycopg2://{username}:{password}@localhost/{db_name}".format(**self.db_config)
        except FileNotFoundError:
            print("Warning! Configuration file not found!\n Insert Username and Password...")
            username = input("Inserisci Username")
            password = getpass.getpass("Inserisci Password")
            host = input("Inserisci host")
            db_name = "covid"
            self.db_config = {"username":username, "password": password, "db_name":db_name, "host":host}
            self.db_string = "postgresql+psycopg2://{username}:{password}@{host}/{db_name}".format(**self.db_config)
            print(self.db_string.replace(password, "*****"))
        finally:
            self.db = create_engine(self.db_string)
            if not database_exists(self.db.url):
                create_database(self.db.url)
                print('Information: "covid" database doesn\'t exist and has been created!')
            print("Succeffully connected to DB!")
    
    def save_db_config(self):
        '''
        Punto 2. Funzionalità per salvare permanentemente la configurazione su 
        un file db_config.json.
        '''
        print("Saving DB configuration file")
        with open('db_config.json', 'w') as fp:
            json.dump(self.db_config, fp)
            print("Configuration file for DB saved as db_config.json")

    def save_analysis(self, df_analysis):
        '''
        Punto 5b (lato DB management): Metodo che permette il salvataggio dei risultati 
        dell'analisi sul DB Postgresql. Richiede il dataframe dell'analisi da salvare generato
        dalla classe "CovidDataHandler" con il metodo "analize_data_subset"
        '''
        with self.db.connect() as connection:
	        df_analysis.to_sql('analysis', con=connection, if_exists='append', 
                                dtype={'denominazione_regione':String,
                                        'mean': Float,
                                        'median': Float, 
                                        'skew':Float,
                                        'max':Float,
                                        'min':Float,
                                        'start_date_selected':String,
                                        'end_date_selected':String,
                                        'analysis_date':DateTime,
                                        'feature_name':String})
	        connection.close()
	        self.db.dispose()
        
    def read_analysis_from_db(self):
        '''
        Metodo per leggere i risultati delle analisi salvate in precedenza nella
        tabella "analysis" del DB Postgres.
        '''
        with self.db.connect() as connection:
            display(pd.read_sql_table("analysis", con=connection))
            connection.close()
            self.db.dispose()
            print("Analysis table correctly loaded.")
