# Predicting Loan Default
## Stage: Problem Framing & Scoping

### Problem Statement
Financial institutions like banks and other lending organizations face significant risk by lending to borrowers who default on their loans, leading to revenue loss. Predicting loan default in advance can allow organizations to minimize these losses. A well-performing model can detect these risky applicants, thus guiding lenders.

### Stakeholder & User
**Stakeholders:** credit risk management team working at the financial institutions  
**Users:** loan underwriters, who may use it as a part of an automated loan approval system  
These predictions will be a part of the decision-making workflow, and must be available at the time of application review.

### Useful Answer & Decision
**Type:** predictive modelling  
**Key metric:** F1 score  
**Artifact:** A trained ML model that will give a default probability score, which will be converted into a binary label- will default or won't default

### Assumptions & Constraints
- **Data Availability:** Data source - Kaggle [https://www.kaggle.com/datasets/urstrulyvikas/lending-club-loan-data-analysis](https://www.kaggle.com/datasets/urstrulyvikas/lending-club-loan-data-analysis)
- **Resource constraints:** 
    + There are sufficient resources for training, and fast inferencing.
    + There are sufficient resources and pipelines in place for retraining every 2-3 months to capture changing trends.
- **Assumptions:**
    + The data accurately represents real world patterns.
    + The data doesn't indirectly incorporate biases.

### Known Unknowns / Risks
- Data drift due to change in borrower behavior or economical conditions
- Since loan default is less common, the data is highly imbalanced, which can affect model performance.
- Transparency and explainability of model predictions
- Potential bias

### Lifecycle Mapping
Goal → Stage → Deliverable
- Reduce loan default risk → Problem Framing & Scoping (Stage 01) → Requirements and challenges

### Repo plan
- `/data/` → Raw and processed datasets
    + should be updated every 2-3 months to capture latest trends
- `/src/` → Core code: data preprocessing, feature engineering, model training, evaluation, and deployment utilities
    + should be updated when code changes
- `/notebooks/` → Exploratory analysis and prototyping
- `/docs/` → Project documentation: repo usage, model design, assumptions
    + should be updated appropriately as changes are made

## Stage: Data Storage
- project/  
    + data/
        - raw
        - processed  

`data/raw` contains the raw and unchanged data.  
`data/processed` contains the data files which are changed by preprocessing, feature engineering etc.

