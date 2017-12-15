# Precision-Prognosis
Insight data science capstone project

www.precision-prognosis.info

#### Motivation: <a id='cell1'></a>

Cancer is a big burden for our society. According to the projections from cancer.gov, there will be 50% more cases in the world in 2030. In the US, we are projected to spend 200 billion dollars  on cancer care in the same year. One of the promising ways to solve this problem is called precision medicine. The idea is that we can further individualize treatments according to their genotype, gene expression and biomarkers.In this project I focused on prognosis part of the precision medicine. 

This is an important issue because many medical decisions benefit from an accurate prognosis. For example, female breast cancer patients may need to delay their treatment for fertility preservation. They want to know the associate risk with that. As an another example, what is the benefit of having additional treatment X ? For risky treatments such as chemotherapy doctors need to know how much increase in survival their patients will get in order to balance the risks and the benefits of the treatment.

To individualize prognosis for each patient, I used their gene expression profile and clinical information. After feature selection, I used a specific type of regression (COX) to build a predictive model. Finally using this model I produced survival predictions, and calculated the benefit of the adjuvant chemotherapy to make treatment recommendations.

I obtained the data by combining multiple files that I gathered through Genomic Data Commons API. I have 1098 patients, and I have the gene  expression levels for 17000 genes. Whether chemotherapy is given to the patients, and their cancer stage are clinical features, the number of days they lived after the diagnosis and vitality are labels.

For feature selection I divided patients into two groups along a cut-off value that maximizes the log-rank statistic. Then, I ranked p-values and corrected for multiple hypothesis testing. What I am doing essentially is trying to find genes that have the maximum survival difference between high and low groups. This procedure identified prognostic genes and reduced the number of features to about 3000.

Next to build my model, I used COX proportional hazards regression. This technique is used in clinical research to relate risk factors or exposures to survival time. After checking the model assumptions, a stepwise procedure is used for further feature selection. Even though, I had 3000 genes, my model didn’t significantly change after taking into account top 50 genes, so I didn’t consider the rest.

And to validate my model I used concordance index (c-index) which is the fraction of pairs where the observation with the higher survival time have the higher probability of survival predicted by your model. 
So the model only using clinical information or genes had an accuracy of 0.70, but combining both genes and clinical features in my model I was able to get c-index of 0.78 (in the validation data). Therefore by using information from patient’s gene expression profile I was able to increase the accuracy of my survival predictions. 

References:

1. http://science.sciencemag.org/content/357/6352/eaan2507

2. http://www.cancer.gov

### Data:

A large file (`patient_df.txt`) containing gene expression data can be downloaded here:

https://drive.google.com/open?id=1RauiKda6SnJKFRqhed-cB2z7Av3Z0--w

#### Requirements:
- Python=3.5
- Numpy
- Pandas
- Bokeh
- Lifelines
- Scipy
- Seaborn
- Matplotlib

#### How to run the Code:
Simply run `Breast_cancer_chemo.ipynb`
