def plot_roc(model, X_test, y_test):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
    
    """
    Plots ROC curves for several models.
    
    Parameters:
    - models: list of models to be evaluated (trained)
    - model_names: list of model names (must be the same length as 'models')
    - X_test: test data (features)
    - y_test: test labels (targets)
        
    """
    
    
    
    # Get predicted class probabilities for the test set 
    y_pred_prob = model.predict_proba(X_test)[:, 1] 
    # Compute the false positive rate (FPR) 
    # and true positive rate (TPR) for different classification thresholds 
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob, pos_label=1)
    # Compute the ROC AUC score 
    roc_auc = roc_auc_score(y_test, y_pred_prob) 
    
    # Plot the ROC curve 
    plt.plot(fpr, tpr, color="orange", label='ROC curve (area = %0.2f)' % roc_auc,) 
    # roc curve for tpr = fpr 
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier', color="green") 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    plt.title('ROC Curve') 
    plt.legend(loc="lower right") 
    plt.show()
    
