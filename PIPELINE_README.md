# Pipeline README

## Purpose
This pipeline converts the raw student depression dataset into model-ready features in a reproducible way. It ensures that preprocessing is applied consistently across training and test data and helps prevent data leakage.

## Main steps
1. Clean the raw dataset:
   - strip column names
   - replace placeholder missing values such as `?` with `NaN`
   - drop the `id` column
   - convert numeric fields to numeric dtype

2. Select the final feature set:
   - numeric features: Academic Pressure, Financial Stress, Age, Work/Study Hours, Study Satisfaction, CGPA
   - categorical features: Dietary Habits, Sleep Duration, Have you ever had suicidal thoughts ?

3. Apply preprocessing:
   - numeric features: median imputation + standard scaling
   - categorical features: constant imputation with `Unknown` + one-hot encoding

4. Split the data:
   - 80/20 train-test split
   - stratified by the target variable `Depression`

## Rationale
- `id` was removed because it has no predictive meaning.
- Median imputation was used for numeric columns because it is robust to skew and outliers.
- One-hot encoding was used for nominal categorical variables.
- `handle_unknown="ignore"` was used so unseen categories in test data do not cause errors.
- Scaling was applied to numeric features because models such as logistic regression are sensitive to feature scale.
- Stratified splitting was used because the target classes are moderately imbalanced.

## Ethical note
The feature `Have you ever had suicidal thoughts ?` is highly predictive but also sensitive and closely related to the target. The pipeline supports sensitivity analysis by allowing this feature to be included or excluded.

## Files
- `src/pipeline.py`: reusable preprocessing and split logic
- notebook: imports the pipeline and uses the same preprocessing consistently
