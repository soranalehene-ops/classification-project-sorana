# Feature Definitions

## Automated Feature Engineering Review

Automated feature generation was explored using FeatureTools with `max_depth=2` on the student depression dataset. Since the dataset is a **single flat table** rather than a multi-table relational dataset, the generated features mainly consisted of **transformation and interaction features** rather than aggregation features. This is still useful for identifying candidate interactions between academic, lifestyle, and mental health related variables. Still, all generated features must be reviewed carefully to avoid noise, redundancy, or target leakage.

The goal of this review was not to keep as many features as possible, but to retain only those that are:
- meaningful in the mental health context,
- unlikely to introduce leakage,
- interpretable enough to explain to stakeholders,
- and plausible for improving classification performance.

## Feature generation setup

- Tool: FeatureTools
- Dataset type: Single-table tabular dataset
- Target: `Depression`
- Generation depth: `max_depth=2`
- Expected output type: Transformation and interaction features
- Review criteria: predictive plausibility, interpretability, leakage risk, ethical sensitivity 

## Retained automated features for further evaluation

### 1. `Academic Pressure * Financial Stress`
Status: Retained for deeper evaluation  
Rationale: EDA showed that both Academic Pressure and Financial Stress are among the strongest predictors of depression. Their interaction may capture a compounding effect, where high academic pressure combined with high financial stress is more predictive than either variable alone.
Expected value: may improve performance by representing joint burden rather than isolated stressors.

### 2. `Academic Pressure / Study Satisfaction`
Status: Retained for deeper evaluation  
Rationale: This feature may capture the imbalance between pressure and perceived satisfaction with studies. A student with high pressure and low satisfaction may be at greater mental health risk than one experiencing high pressure but still feeling satisfied and engaged.
Expected value: Could represent a more psychologically meaningful stress profile than either variable alone.

### 3. `Work/Study Hours * Academic Pressure`
Status: Retained for deeper evaluation  
Rationale: EDA suggested that Work/Study Hours and Academic Pressure are both associated with the target. Their interaction may better represent workload intensity than either feature by itself.
Expected value: may improve classification by capturing students who both work long hours and feel highly pressured.

### 4. `Age * Academic Pressure`
Status: Retained for deeper evaluation  
Rationale: Age showed a moderate relationship with the target in EDA. An interaction between **Age** and **Academic Pressure** may reflect that pressure affects younger and older students differently.
Expected value: Could help capture subgroup differences within the student population.

### 5. `Financial Stress / Work/Study Hours`
Status: Retained for deeper evaluation  
Rationale: This feature may express whether financial stress is high relative to reported workload. In some cases, financial stress may remain elevated even when study hours are low, which could indicate a different risk profile than purely academic overload.
Expected value: may provide a more nuanced view of stress burden.

## Discarded automated features

### 1. Features involving the target variable `Depression`
Status: Discarded  
Rationale:Any generated feature derived directly or indirectly from the target would introduce **data leakage** and make the model invalid.

### 2. Features built from `id`
Status: Discarded  
Rationale: `id` column is only an identifier and has no real-world predictive meaning. Any automated feature based on it would add noise rather than useful signal.

### 3. Highly complex city-based interactions
Examples: interactions combining `City` with multiple other variables  
Status: Discarded for now  
Rationale: `City` has high cardinality, and many generated combinations are likely to create sparse, unstable, and hard-to-interpret features. These could also reduce generalizability.

### 4. Profession-based automated interactions
Status: Discarded for now  
Rationale:`Profession` variable is almost constant in this dataset, with nearly all records labeled as **Student**. Automated interactions using this feature are unlikely to add value.


### 5. Redundant arithmetic combinations with weak base features
Examples: combinations involving `Job Satisfaction`, `Work Pressure`, or very weak numerical relationships  
Status: Discarded for now  
Rationale: EDA showed that some variables had very weak direct relationships with the target. Automatically generated combinations using these weak features are unlikely to improve performance and may increase noise.

## Features requiring ethical caution

### `Have you ever had suicidal thoughts ?`
Status: Not used for automated interaction generation at this stage  
Rationale:the feature was one of the strongest predictors in EDA, but it is also ethically sensitive and closely related to the target concept. Automatically generating multiple interaction features from it may create a model that relies too heavily on one target-proximal variable.
Decision: models should be compared **with and without** this variable before deciding whether to include it in more advanced feature engineering.

## Final review decision

The automated feature generation step was useful for identifying a small set of plausible interaction features, especially around:
- academic burden,
- financial stress,
- satisfaction,
- and workload.

However, because the dataset is a single-table student mental health dataset, automated feature generation did not replace manual review. Only a limited subset of generated features was considered appropriate for deeper testing. Features with high leakage risk, low interpretability, or low expected value were excluded.

## Next action

The retained automated features will be tested in the modelling pipeline and compared against the baseline feature set using:
- accuracy,
- precision,
- recall,
- F1-score,
- and ROC-AUC.

Features that improve performance without harming interpretability or ethical validity will be kept in the final pipeline.
