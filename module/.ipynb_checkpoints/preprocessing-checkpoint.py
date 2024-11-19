# module preprocessing

def PCA_model(data,n_components=4,columns=['Composante 1', 'Composante 2','Composante 3','Composante 4']):

    """
    This function is to compute PCA algorithm.
    data : pd.DataFrame
    n_componwnts (int) : number of principal component analysis
    columns (list) : list of name of your components
    
    return : components data frame and explained variance
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    scaler = StandardScaler()
    df_standardized = scaler.fit_transform(data)
    
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df_standardized)
    
    
    pca_df = pd.DataFrame(data=pca_result, columns=columns)

    PC_values = np.arange(pca.n_components_) + 1
    plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.show()
        
    return (pca_df, pca.explained_variance_ratio_)





    
def separation_of_train_test(data,label_name,size_=0.3):
    """
    This function is to separate a test_set and the train_set.
    data :pd.DataFrame
    label_name : str
    size_ : pourcentage of the test set
    
    return : x_tain,x_test and y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    seed=1234
    y=data[label_name]
    x=data.drop(label_name,axis=1)
    X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=size_,random_state=seed,stratify=y)
    return X_train,X_test,y_train,y_test



def to_standardized(train_data):
    """
    This function allow you to standardized your data.
    return : data_standardized and the scaler to scale your test data
    """
    from sklearn.preprocessing import MinMaxScaler
    import pandas as pd
    scaler=MinMaxScaler()
    scaler.fit(train_data)
    return pd.DataFrame(scaler.transform(train_data),columns=train_data.columns),scaler


def features_selection(x_train,x_test,y_train,seuil=0.03,seed=1234):
    """
    This function purpose is to select the most important var
    x_train : pd.DataFrame
    x_test : pd.DataFrame
    seuil : threshold of importance of var
    return x_train,x_test
    """
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    rdf=RandomForestClassifier(random_state=seed)
    rdf.fit(x_train,y_train)
    plt.figure()
    features_important=pd.Series(rdf.feature_importances_, index=x_train.columns).sort_values(ascending=False)
    sns.barplot(x=features_important.index, y=features_important)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Variables")
    plt.ylabel("score of importance")
    plt.title("Importance of feature")
    plt.show()
    features_selected=features_important[features_important>seuil].index.to_list()
    x_train=x_train[features_selected]
    x_test=x_test[features_selected]
    return x_train,x_test