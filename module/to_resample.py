def resampling_data(x,y,target_name,method="SMOTE"):
    
    seed=1234
    from imblearn.over_sampling import ADASYN, SMOTE
    if method=="SMOTE":
        resampler=SMOTE(sampling_strategy = 'auto' , random_state =seed , k_neighbors = 10)
        resampler.fit(x,y)
        x_resampled,y_resampled=resampler.fit_resample(x,y)
    else :
        resampler=ADASYN(sampling_strategy = 'auto' , random_state =seed , k_neighbors = 10)
        resampler.fit(x,y)
        x_resampled,y_resampled=resampler.fit_resample(x,y)
    if method=="sampling":
        test_data=pd.concat([x,y])
        minority=test_data[test_data.Y1_seuil==1]
        majority=test_data[test_data.Y1_seuil==0]
        minority_upsampled = resample(minority,replace=True,n_samples=len(majority), random_state=seed)
        resampler = pd.concat([majority,minority_upsampled])
        x_resampled= resampler.drop(target_name,axis=1)
        y_resampled= resampler[target_name]
    return x_resampled,y_resampled,resampler
                 