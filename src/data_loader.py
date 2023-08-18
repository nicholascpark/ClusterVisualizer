import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
from io import BytesIO
from azure.storage.blob import ContainerClient
from urllib.parse import urlparse
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imbpipeline
import warnings
warnings.filterwarnings('ignore')

class DataLoader:

    def __init__(self, config):

        self.config = config
        self.preprocessor = None

    def load_data_from_cloud(self, url = None, get_additional = True):

        if url is None:
            url = self.config['data']['data_url'] + self.data_source
        connect_str = self.config['connect_str']
        path = urlparse(url).path.split("/")
        container = self.config['container_name']
        if url.startswith('https'):
            blob_path = '/'.join(path[2:])
        elif url.startswith('abfss'):
            blob_path = '/'.join(path[1:])
        try:
            container_client = ContainerClient.from_connection_string(conn_str=connect_str, container_name=container)
            parquet_blobs = [blob.name for blob in container_client.list_blobs(name_starts_with = blob_path) if blob.name.endswith('.parquet')]
            print('Pulling data from blob path:', blob_path)
            assert len(parquet_blobs) >= 1, 'Cannot have less than 1 parquet blob in specified blob path'
            df = pd.concat(
                        objs = [
                                pd.read_parquet(
                                    BytesIO(container_client.download_blob(parquet_blob).readall()), 
                                    engine='pyarrow'
                                    ) 
                                    for parquet_blob in parquet_blobs
                                ],
                        ignore_index = True
                    )
            BytesIO().seek(0)
        except Exception as e:
            print("Container client failed to generate.")
            print(e)
            raise e
        
        if get_additional:
            df['age_of_buy'] = df["CAMPAIGN_CODE"] - df["GROUP_CODE"]
            if {"ACTUAL_DEBT_VALUE", "ACTUAL_INCOME"} <= set(df.columns):
                df["DTI"] = df["ACTUAL_DEBT_VALUE"] / df["ACTUAL_INCOME"] 
                df["DTI"] = df["DTI"].apply(lambda x: 1.1 if x > 1.1 else x)
        print(df.columns)
        print(df.shape)

        df = df[df['CAMPAIGN_CODE'] < self.config['max_campaign']]
        return df

    def extract_data(self, path = None, source_type = 'pickle', fraction = 0.1):

        if path is not None:
            if path.endswith('.pkl'):
                data = pd.read_pickle(path)
                data["DTI"] = data["DTI"].apply(lambda x: 1.1 if x > 2 else x)
            elif path.endswith('.csv'):
                data = pd.read_csv(path)
        else:
            if source_type == 'pickle':
                data = pd.read_pickle(self.config['data_path']['input_pkl'])
                data["DTI"] = data["DTI"].apply(lambda x: 1.1 if x > 2 else x)
            elif source_type == 'csv':
                data = pd.read_csv(self.config['data_path']['input_csv'])
            else:
                data = self.load_data_from_cloud(self.config['data_path']['input_url'])

        data = data.rename(columns = lambda x: x.upper())

        # print("1", data.columns)

        def NestedDictValues(d):
            for v in d.values():
                if isinstance(v, dict):
                    yield from NestedDictValues(v)
                else:
                    yield v

        data = data[sum(NestedDictValues(self.config['columns']), [])]
        print(".\n.\n.\n... Data loaded successfully with shape: ", data.shape)

        # print("2", data.columns)

        # Sample the data
        data = data.sample(frac = float(fraction), random_state=42)
        print("Data sampled with fraction:", fraction, "and shape: ", data.shape)

        # print("3", data.columns)

        data = self.preprocess_data(data, self.config['columns']['features'])

        # print(data.columns)

        return data

    def preprocess_data(self, data, features_config):

        # Perform categorical encoding
        categorical_features, numeric_features = features_config['categorical'], features_config['numeric']

        numeric_transformer = Pipeline(
            steps = [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ],
        )
        categorical_transformer = Pipeline(
            steps = [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("One-Hot Encode", OneHotEncoder(handle_unknown="ignore", drop = "first")),
                ("scaler", StandardScaler(with_mean=False))
            ]
        )
        self.preprocessor = ColumnTransformer(
            transformers = [
                ("Num", numeric_transformer, numeric_features),
                ("Cat", categorical_transformer, categorical_features),
            ],
            remainder='passthrough'
        )

        original_data = data[categorical_features + numeric_features].rename(str.lower, axis='columns').reset_index(drop=True)

        processed_data = self.preprocessor.fit_transform(data)#, data_fixed['Response'])
        feature_names_out = [x.split('__')[-1] for x in self.preprocessor.get_feature_names_out()]
        processed_data = pd.DataFrame(processed_data, columns = feature_names_out)
        processed_data[original_data.columns] = original_data.values
        processed_data['Response'] = data['RESPONSE']

        # processed_data = processed_data.sort_values('GROUP_CODE', ascending = False).drop_duplicates('CUSTOMER_ID',keep='first')

        print('Preprocessing data complete with shape:, ', processed_data.shape)

        return  processed_data