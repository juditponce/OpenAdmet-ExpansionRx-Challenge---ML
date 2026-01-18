# OpenAdmet-ExpansionRx-Challenge---ML
Machine Learning model for ADMET property prediction

## Authors: 
- Judit Ponce Casañas
- Alberto Alexander Robles Loaiza

## Model Description
A supervised machine learning pipeline was developed to predict multiple ADMET-related properties from molecular structure information. Independent regression models were trained for each ADMET endpoint, reflecting the heterogeneous nature of ADMET properties and their experimental variability.

An initial baseline model based on Support Vector Machines (SVM) was implemented. Early experiments revealed strong overfitting behavior, motivating the inclusion of additional non-linear models with improved stability, namely Random Forest (RF) and Extreme Gradient Boosting (XGBoost). These models were selected due to their robustness and widespread use in ADMET and QSAR modeling.

All models take descriptor-based representations (Mordred) and molecular fingerprints as input and output continuous predictions corresponding to experimental ADMET measurements.


## ADMET data analysis and transformation
Prior to model training, the ADMET variables were analyzed individually to assess their distributions and numerical ranges. Given the typically skewed nature of experimental ADMET measurements, all target variables were transformed to a logarithmic scale when appropriate.

This transformation was applied before any model training or evaluation steps and aimed to:
- Reduce distributional skewness  
- Stabilize variance  
- Improve model convergence and predictive performance 

## Molecular Representation
Molecular structures were encoded using descriptor- and fingerprint-based representations computed from SMILES strings. Two types of molecular representations were employed:

- **Mordred descriptors**, capturing a wide range of physicochemical, topological, and structural features  
- **Molecular fingerprints**, providing fixed-length binary or count-based representations of molecular substructures  

Prior to training, descriptor matrices were preprocessed by removing constant and duplicated features. Descriptors containing missing values were removed, and the same feature selection was consistently applied to both training and test sets to ensure complete input matrices for each ADMET endpoint. Feature scaling was applied when required by the learning algorithm.

## Training Strategy
Model training was performed independently for each ADMET endpoint. For a given property, only molecules with available experimental values were included, avoiding artificial imputation of missing data.

A model comparison strategy was adopted in which multiple learning algorithms (SVM, Random Forest, and XGBoost) were evaluated under identical conditions. For each endpoint and model, training was carried out using a pipeline-based approach to ensure consistent preprocessing and to prevent data leakage.

Each pipeline included variance filtering, correlation-based feature selection (using a Spearman correlation threshold), feature scaling, and model fitting. Feature selection steps were learned exclusively from the training data and then applied consistently to validation and test data.

Hyperparameter optimization was performed using grid search combined with 5-fold cross-validation. Model performance was assessed using cross-validated predictions on the training set, allowing for robust estimation of generalization performance while reducing sensitivity to a single data partition.

This strategy enabled direct comparison of model stability and performance across different algorithms and endpoints. Performance distributions across models and endpoints can be summarized using boxplots, providing an evaluation independent of a single train–test split.

No external pre-training on additional datasets was performed.

## Additional optimization steps
Hyperparameters were optimized empirically for each model type to balance predictive performance and generalization. Particular attention was paid to mitigating overfitting, especially in early SVM models, which guided the selection of more stable ensemble-based approaches.

## Performance Comments
Initial experiments using SVM models highlighted substantial overfitting, characterized by strong training performance but reduced generalization on held-out data. This observation motivated the inclusion of Random Forest and XGBoost models, which demonstrated improved robustness and more stable performance across different data splits.

Across endpoints, ensemble-based models generally showed lower variance in performance metrics and better generalization behavior. While absolute performance varied depending on the ADMET property, correlation-based metrics consistently indicated that the models were able to capture meaningful structure–property relationships.


