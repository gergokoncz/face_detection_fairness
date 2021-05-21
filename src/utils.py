import asyncio
from pathlib import Path
import pandas as pd
from os import listdir

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from typing import Dict, List, Iterator

def parse_creds() -> Dict[str, str]:
    creds_dict: Dict = dict()
    creds_path: Path = Path.cwd().parent.joinpath('creds').joinpath('azure_creds.txt')
    with open(creds_path, 'r') as f:
        creds_dict['endpoint'] = f.readline().strip()
        creds_dict['region'] = f.readline().strip()
        creds_dict['key'] = f.readline().strip()
    return creds_dict
        

def get_visionclient() -> ComputerVisionClient:
    creds: Dict[str, str] = parse_creds()
    vision_client : ComputerVisionClient = ComputerVisionClient(creds.get('endpoint'), CognitiveServicesCredentials(creds.get('key')))
    return vision_client

def get_valdf() -> pd.DataFrame:
    path: Path = Path.cwd().parent.joinpath('data').joinpath('fairface_label_val.csv')
    return pd.read_csv(path)

def get_azuredf() -> pd.DataFrame:
    path: Path = Path.cwd().parent.joinpath('data').joinpath('azure_analyzed.csv')
    return pd.read_csv(path)

def get_azured_val() -> pd.DataFrame:
    path: Path = Path.cwd().parent.joinpath('data').joinpath('azure_val_merged.csv')
    return pd.read_csv(path)

def get_images() -> List[Path]:
    folder_path: Path = Path.cwd().parent.joinpath('data').joinpath('val')
    return (folder_path.joinpath(x) for x in listdir(folder_path))

if __name__ == '__main__':
    print(get_visionclient())
    #print(get_val_file().head())
    print(len(list(get_images())))