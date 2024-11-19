def import_data(path):
    """
    This function import the data. You just need to enter the path
    path : str
    return : DataFrame
    """
    import pandas as pd
    data=pd.read_csv(path)
    return data
