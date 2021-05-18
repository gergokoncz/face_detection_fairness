import asyncio
from pathlib import Path
import pandas as pd

from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials

from typing import Dict

def parse_creds() -> Dict[str, str]:
    creds_dict: Dict = dict()
    creds_path: Path = Path.cwd().parent.joinpath('creds').joinpath('azure_creds.txt')
    with open(creds_path, 'r') as f:
        creds_dict['endpoint'] = f.readline().strip()
        creds_dict['region'] = f.readline().strip()
        creds_dict['key'] = f.readline().strip()
    return creds_dict
        

def get_faceclient() -> FaceClient:
    creds: Dict[str, str] = parse_creds()
    face_client : FaceClient = FaceClient(creds.get('endpoint'), CognitiveServicesCredentials(creds.get('key')))
    return face_client

def get_val_file() -> pd.DataFrame:
    path: Path = Path.cwd().parent.joinpath('data').joinpath('fairface_label_val.csv')
    return pd.read_csv(path)

if __name__ == '__main__':
    #print(get_faceclient())
    #print(get_val_file().head())