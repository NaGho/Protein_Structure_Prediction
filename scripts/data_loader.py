import pandas as pd

def load_cb513_dataset(file_path: str):
    """
    Loads the CB513 dataset from a CSV file.
    
    Parameters:
        file_path (str): Path to the CB513 dataset CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing the dataset.
    """
    data = pd.read_csv(file_path)
    return data
