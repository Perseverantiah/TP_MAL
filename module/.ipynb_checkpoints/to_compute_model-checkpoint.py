def compute_model(model_name):
    """
    This function allows you to compute different sort of model.
    You have RandomForest,XGBoost
    model_name : (strg) The possible value of model_name is "rdf" for RandomForest, "xgb" for XGBoost,
    
    x_train : pd.DataFrame
    y_train : Series
    
    return model
    """

    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    seed=1234
    
    if model_name=="rdf":
       model=RandomForestClassifier(n_estimators=30,max_depth=5,random_state=seed)
   

    if model_name == "Naive" :
        model = GaussianNB()
        
    if model_name == "xgboost":
        model = XGBClassifier(n_estimators=10, max_depth=3, use_label_encoder=False, eval_metric='logloss')
        

    if model_name=="Decision Tree":
        model=DecisionTreeClassifier(min_samples_split=20, min_samples_leaf=10, random_state=1234,ccp_alpha=0.01)
        
    if model_name=="logistic":
        model=LogisticRegression (random_state=seed, max_iter=500)
            
    return model
