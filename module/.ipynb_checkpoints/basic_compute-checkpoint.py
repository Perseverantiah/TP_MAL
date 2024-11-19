def basic_compute(data):
    """
    This function is void function.
    data : pd.DataFrame

    It checks the size of the data set, the type of var and the quantiles of each var.

    """

    
    print("The shape of your dataset {}".format(data.shape))
    print("----------------------------------------------------------------------------------------------")
    print("Type of var")
    print(data.info())
    print("----------------------------------------------------------------------------------------------")
    print("Quantile of each var")
    print(data.describe())
    print("----------------------------------------------------------------------------------------------")
    
    