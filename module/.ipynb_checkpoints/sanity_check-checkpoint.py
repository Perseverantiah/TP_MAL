def sanity_check(data):
    """
    This function check the sanity of the data. We check that we don't have a missing vallue
    data : pd.DataFrame
    return : DataFrame
    """
    if (data.isna().sum()).sum()>0 :
        print("Missing values found. Dropping missing values.")
        data=data.dropna()
    else :
        print("No missing values found.")

    if (data.duplicated().sum()).sum()>0:
        print("There are a duplicated individuals in our data")
    else :
        print("No missing duplicated values found.")
    
    return data