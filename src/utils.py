from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import k_fold_cross_validation,concordance_index
from lifelines.statistics import logrank_test
import pandas as pd
import numpy as np

def COXPH_backward_elimination(COX_dataset,penalizer=0):
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(COX_dataset , duration_col='Days', event_col='Vitality')
    
    
    for i in range(COX_dataset.shape[1]-2):
        current_ps = cph.summary['p']
        highest_factor = current_ps.idxmax()
        highest_p = current_ps.max()
        if highest_p<0.05:
            break
        
        COX_dataset = COX_dataset.drop(highest_factor,axis=1)
        cph.fit(COX_dataset, duration_col='Days', event_col='Vitality')

    return cph


def features_vs_cindex(training_data,validation_data,gene_list):
    features_cindex = []
    for i in range(1,len(gene_list)+1):
            # Training
            COX_dataset = training_data[gene_list[:i]+['Days','Vitality']]
            cph = COXPH_backward_elimination(COX_dataset,penalizer=0)
            final_factors = cph.summary.index.tolist()
            train_c_index = concordance_index(COX_dataset['Days'],
                            -cph.predict_partial_hazard(COX_dataset[final_factors]).values.ravel(),
                                        COX_dataset['Vitality'])
            # Validation
            COX_dataset = validation_data
            valid_c_index = concordance_index(COX_dataset['Days'],
                            -cph.predict_partial_hazard(COX_dataset[final_factors]).values.ravel(),
                                        COX_dataset['Vitality'])
            
            features_cindex.append([i,valid_c_index,train_c_index])
    return np.array(features_cindex)

def regularization_vs_cindex(training_data,validation_data,param_list,number_of_features,gene_list):
    penalty_cindex = []
    for penalty in param_list:
            # Training
            COX_dataset = training_data[gene_list[:number_of_features]+['Days','Vitality']]
            cph = COXPH_backward_elimination(COX_dataset,penalizer=penalty)
            final_factors = cph.summary.index.tolist()
            train_c_index = concordance_index(COX_dataset['Days'],
                            -cph.predict_partial_hazard(COX_dataset[final_factors]).values.ravel(),
                                        COX_dataset['Vitality'])
            # Validation
            COX_dataset = validation_data
            valid_c_index = concordance_index(COX_dataset['Days'],
                            -cph.predict_partial_hazard(COX_dataset[final_factors]).values.ravel(),
                                        COX_dataset['Vitality'])
            
            penalty_cindex.append([penalty,valid_c_index,train_c_index])
    return np.array(penalty_cindex)
