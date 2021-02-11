# Capstone Project

[YouTube!](https://www.youtube.com/watch?v=5mdBlyQqLQ8)

## Project Overview

This is the capstone project as part of Azure Nanodegree Program, Machine Learning Engineer with Microsoft Azure.

As required by the project we have selected an external dataset "Heart failure clinical records" (detail section below) and performed machine learning task of predicting survival of patients facing heart failure (binary classification ) using 12 feautures identified in the data set. 

As per the project guidance we created two experiments, one using Automated ML (AutoML) and another model using Scikit Learn Logistic Regression classifier, whose hyperparameters were tuned using HyperDrive

From each experiment we selected one best model based on "accuracy" metric. 

Details are: 

i) AutoML best model: 

From AutoML experiment best model selected was VotingEnsemble with 88% accuracy.

ii) HyperDrive experiment

From Sci-kit Learn trained model , tuned with Hyperdrive, best model was  Logistic Regression with 83 % accuracy


Based on comparison of two models we selected VotingEnsemble model from AutoML eperiment and then deployed it as a webservice (REST API). We then tested the webservice by sending a request to the model endpint.



## Dataset used

We have used Heart failure clinical records data set as published on "UCI Machine Learning Repository"and "Kaggle Heart Failure Prediction" competion. 

This data set has been published as part of folowing paper: Davide Chicco, Giuseppe Jurman: "Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone". BMC Medical Informatics and Decision Making 20, 16 (2020).

This dataset contains the medical records of 299 heart failure patients collected at the Faisalabad Institute of Cardiology and at the Allied Hospital in Faisalabad (Punjab, Pakistan), during April–December 2015 [52, 66]. The patients consisted of 105 women and 194 men, and their ages range between 40 and 95 years old (Table 1). All 299 patients had left ventricular systolic dysfunction and had previous heart failures that put them in classes III or IV of New York Heart Association (NYHA) classification of the stages of heart failure.

The dataset contains 13 features, which report clinical, body, and lifestyle information, that we briefly describe here.

- Age: age of the patient (years)
- Anaemia: decrease of red blood cells or hemoglobin (boolean)
- High blood pressure: if the patient has hypertension (boolean)
- Ceatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L)
- Diabetes: if the patient has diabetes (boolean)
- Ejection fraction: percentage of blood leaving the heart at each contraction (percentage)
- Platelets: platelets in the blood (kiloplatelets/mL)
- Sex: woman or man (binary)
- Serum creatinine: level of serum creatinine in the blood (mg/dL)
- Serum sodium: level of serum sodium in the blood (mEq/L)
- Smoking: if the patient smokes or not (boolean)
- Time: follow-up period (days)
- [target] Death event: if the patient deceased during the follow-up period (boolean)


## Data import into Azure ML Studio workspace.

We have source data set as a csv file from kaggle competion [Kaggle!](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data). 

We uploaded this file into our workspace and then uploaded it into our deafult datastore. From their convert this file into Tabular data set for our AUtoML eperiment purposes.

For our hyperdrive expermient, we read data as a pandas dataframe into our experiment.

## AutoML experiment: settings and configuration

AutoML experiment was configured using folloing setting:

- 4 concurrent iterations of models.
- primary metric as "accuracy".
- machine learning task as "Classification". 
- Automatic feature selection by AutoML.
- Early stopping enabled.


## Hyperparameter search: types of parameters and their ranges

As machine learning task at hand was of Classification, we use Sci-kit Learn Logistic Regression classifier. 
We decided to tune two hyperparameters as follows: 

i) C: This controls regularization in model i.e co-efficient values. It is inverse of regularization strength  i.e smaller values cuase stronger regularization. We tested it using Hyperdrive using a unform sample space from 0 to 1.

ii) max_iter: Maximum number of iterations allowed for model to converge. We tested it using Hyperdrive using a set sample of list values 50,100,150,200,250.

Hyperdrive parameter sampler was RamdomParameterSampling. This rendomly selects paramter values from sample space provided. 

Hyperdrive was configured to select best parameters using highest accuracy scored produced and the goal was to maximize the accuracy metric. A total of 40 model runs were alowed with 4 max concurrent runs. 


## Two models with the best parameters

i) From AutoML experiment best model selected was VotingEnsemble with 88% accuracy. Details of its parameters are as follows:
- min_child_weight=1,
- missing=nan,
- n_estimators=10,
- n_jobs=1,
- nthread=None,
- objective='reg:logistic',
- random_state=0,
- reg_alpha=0,
- reg_lambda=0.625,
- scale_pos_weight=1,
- seed=None,
- silent=None,
- subsample=1,
- tree_method='auto', 
- flatten_transform=None,
- weights=[0.125, 0.125, 0.125,  0.125, 0.125, 0.125, 0.125, 0.125]
       
         
   Run(Experiment: capstone-Automl,
Id: AutoML_8934d6c4-8831-4d4b-98e5-907b3bdab98d_40,
Type: azureml.scriptrun,
Status: Completed)

{'precision_score_weighted': 0.8958617020161247, 'average_precision_score_micro': 0.9269018734864382, 'AUC_weighted': 0.9191387579070647, 'balanced_accuracy': 0.8511186336229242, 'matthews_correlation': 0.7423301682990012, 'f1_score_weighted': 0.8784478545347623, 'precision_score_macro': 0.8941003107622102, 'precision_score_micro': 0.8831034482758622, 'f1_score_macro': 0.8596588655909961, 'recall_score_weighted': 0.8831034482758622, 'f1_score_micro': 0.8831034482758622, 'weighted_accuracy': 0.9035554026334669, 'log_loss': 0.3830698345834191, 'recall_score_macro': 0.8511186336229242, 'AUC_macro': 0.9191387579070645, 'norm_macro_recall': 0.7022372672458486, 'average_precision_score_weighted': 0.9299034435441277, 'accuracy': 0.8831034482758622, 'average_precision_score_macro': 0.9077163867341911, 'recall_score_micro': 0.8831034482758622, 'AUC_micro': 0.9237253269916765, 'confusion_matrix': 'aml://artifactId/ExperimentRun/dcid.AutoML_8934d6c4-8831-4d4b-98e5-907b3bdab98d_40/confusion_matrix', 'accuracy_table': 'aml://artifactId/ExperimentRun/dcid.AutoML_8934d6c4-8831-4d4b-98e5-907b3bdab98d_40/accuracy_table'}


ii) From Sci-kit Learn trained model , tuned with Hyperdrive, best model was  Logistic Regression with 83 % accuracy. Details of its parameters are as follows:

['--C', '0.8848572144734638', '--max_iter', '100']


## Deployed model and instructions on how to query the endpoint with a sample input

Based on higher accuracy metric produced, we selected VotingEnsemble model produced by AutoML experiment for deployment. 
In order to deploy it , we first registerd the model and provided it an environment for deployment. We took advantage of Azure provided environment "AzureML-AutoML" . 
We set up Inference Configuration and provided it with a scoring file, this file contained API model (i.e fields that API would need for data interchange).
We then deployed the model using Azure Container Instance Webservices (Aci). Deployment enabled a REST API that provide scoring uri with keys for authentication. 
We passed test data inform of Json load to webservice configured and it validated our deployment by providing a response in expected format (1,0)


## How to improve the project in the future

We can suggest following improvments that may result in better model or faster model deployment:

i) Use more powerful computer cluster such as GPU instanced with more nodes. This may enable increase in concurrent iterations.

ii) Data has class imbalance with many 1/3 deaths events vs 2/3 non deaths events. We might address it by procuring more data.

iii) We need to assess Classifiers that AutoML had not tested and hyperparameters that were not configured. We might further wish to train our data using those model and parameters by using hyperdrive run and experiment with a different set of parameters.

iv) In Hyperdrive experimnet test might use more classifiers including some of ensemble classifiers as identified by AutoML. This might improve model performance by identifying a faster and more accurate model. 

v) In Hyperdrive experiment use Bayesian Parameter sampling: This might make experiment run faster and be able to quickly identify best hyperparameter.

vii) In Hyperdrive experiment test more hyperparameter for tuning such as penalty, solver, class_weight etc. They might improve model performance by testing a hyperparameter combination that is able to yield more accurate model.

iv) In Hyperdrive experiment to address class imbalance in data by either using SMOTE resampling technique or using class_weight parameter.

vi) In Hyperdrive experiment we have now performed any feature engineering or data standarization/normalization. Conversely we have not performed any Principal Component Analysis (PCA) to identify features with best predictive powers. We might perform these steps/techniques to improve score of model on accuracy metric.

vii) We might select further types of classifiers from Sci-kit learn library like Decision Tree classifier etc and train them to get a model with better accuracy score.


## Screen shots with a short description

i) AutoML Model: 

-screenshot of the RunDetails widget that shows the progress of the training runs of the different experiments.

![](https://github.com/nabeelsana/Capstone_Project/blob/master/starter_file/AutoML_RunWidget1.png)


-screenshot of the best model based in accuracy metric with its run id.


![](https://github.com/nabeelsana/Capstone_Project/blob/master/starter_file/AutoML_BestModel2.png)


![](https://github.com/nabeelsana/Capstone_Project/blob/master/starter_file/AutoML_BestModel3.png)


ii) Hyperdrive Model:

-Screenshot of the RunDetails widget that shows the progress of the training runs of the different experiments.


![](https://github.com/nabeelsana/Capstone_Project/blob/master/starter_file/HyperDrive_Runwidget1.png)


![](https://github.com/nabeelsana/Capstone_Project/blob/master/starter_file/HyperDrive_Runwidget2.png)


![](https://github.com/nabeelsana/Capstone_Project/blob/master/starter_file/HyperDrive_Runwidget3.png)


![](https://github.com/nabeelsana/Capstone_Project/blob/master/starter_file/HyperDrive_Runwidget4.png)


-screenshot of the best model with its run id and the different hyperparameters that were tuned.


![](https://github.com/nabeelsana/Capstone_Project/blob/master/starter_file/HyperDrive_BestModel5.png)



Deploying the Model:

-screenshot showing the deployed model, endpoint as active.


![](https://github.com/nabeelsana/Capstone_Project/blob/master/starter_file/Model_EndPoint1.png)


![](https://github.com/nabeelsana/Capstone_Project/blob/master/starter_file/Model_EndPoint2.png)


-After completion of eperiment, Webservice is being deleted

![](https://github.com/nabeelsana/Capstone_Project/blob/master/starter_file/Webservicedeleted1.png)

