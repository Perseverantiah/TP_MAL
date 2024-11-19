def evaluation_of_model(model,features,label):
    
    """
    This function is to compute the metrics of model and to save it.
    model : the model
    features : features for prediction (pd.DataFrame)
    """
    from sklearn.metrics import classification_report
    print(classification_report(label,model.predict(features)))
    
    