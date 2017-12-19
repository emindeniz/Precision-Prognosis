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

def calculate_cutoffs(patient_df,all_genes_list,quantiles):
    start_time = timeit.default_timer()
    training_cutoffs = []

    for gene in all_genes_list:
        best_value = 0
        best_p = 1

        for quantile in quantiles:
            current_value = patient_df[gene].quantile(quantile)
            results = logrank_test(patient_df.loc[patient_df[gene]>current_value,'Days']/365, 
                                   patient_df.loc[patient_df[gene]<=current_value,'Days']/365, 
                                   patient_df.loc[patient_df[gene]>current_value,'Vitality'], 
                                   patient_df.loc[patient_df[gene]<=current_value,'Vitality'], alpha=.99)
            if results.p_value<best_p:
                best_value = current_value
                best_p = results.p_value
                best_quantile = quantile

        kmf = KaplanMeierFitter()
        kmf.fit(patient_df.loc[patient_df[gene]>best_value,'Days']/365, 
                      patient_df.loc[patient_df[gene]>best_value,'Vitality'])
        higher_median = kmf.median_
        kmf.fit(patient_df.loc[patient_df[gene]<=best_value,'Days']/365, 
                      patient_df.loc[patient_df[gene]<=best_value,'Vitality'])
        lower_median = kmf.median_

        training_cutoffs.append((gene,best_value,best_p,best_quantile,higher_median,lower_median))


    training_cutoffs = pd.DataFrame(training_cutoffs,columns=['Symbols',
                                                             'Expression Cutoffs',
                                                             'log-rank P Values',
                                                             'best_quantile','higher_median','lower_median'])
    print(timeit.default_timer()-start_time)
    return training_cutoffs
