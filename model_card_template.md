# Model Card

For additional information see the [Model Card paper](https://arxiv.org/abs/1810.03993).

## Model Details

Educational Census salary classifier: scikit-learn `RandomForestClassifier` with 100 trees and `random_state=42`. The fitted classifier, `OneHotEncoder`, and `LabelBinarizer` are saved together in `model/artifacts.joblib`. Detailed overall and slice results are written to `model/evaluation_metrics.json`.

## Intended Use

This model is for educational practice in data preprocessing, classification, evaluation, and deployment. It must not be used to make employment, compensation, lending, eligibility, or other decisions about people.

## Training Data

The model uses the project-provided `data/census.csv`; `salary` is the target. Categorical fields are one-hot encoded and numeric fields are passed through. String whitespace is stripped at the field edges. A stratified 80/20 split with `random_state=42` is used, with all preprocessing fit on training data only.

## Evaluation Data

The held-out evaluation set contains 6,513 rows and was not used to fit the model or preprocessing objects. Slice metrics are computed on this same set. The binary positive class is `>50K`.

## Metrics

Held-out overall metrics:

| Metric | Score |
| --- | ---: |
| Precision | 0.7353 |
| Recall | 0.6378 |
| F1 | 0.6831 |

Selected demographic slices (full metrics for all categorical features are in `model/slice_output.txt`):

| Feature | Value | Rows | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| sex | Male | 4,355 | 0.7298 | 0.6432 | 0.6838 |
| sex | Female | 2,158 | 0.7680 | 0.6082 | 0.6788 |
| race | White | 5,533 | 0.7340 | 0.6440 | 0.6861 |
| race | Black | 662 | 0.8060 | 0.5806 | 0.6750 |
| race | Asian-Pac-Islander | 200 | 0.6596 | 0.5962 | 0.6263 |
| race | Amer-Indian-Eskimo | 73 | 0.6667 | 0.4444 | 0.5333 |
| race | Other | 45 | 1.0000 | 0.7500 | 0.8571 |

Groups containing only one target class are marked `single_class` in the slice report. Small group scores are uncertain and should be interpreted cautiously.

## Ethical Considerations

The dataset includes sensitive or protected characteristics, including age, sex, and race, along with socioeconomic and employment information. The model can reproduce biases in its training data. These evaluation results do not establish fairness or suitability for real decisions.

## Caveats and Recommendations

- This is an educational model trained on a supplied historical dataset; it may not represent current populations or conditions.
- Metrics describe one split and may vary on other samples.
- Do not use this model to evaluate an individual's qualifications, worth, or likely compensation.
- Any non-educational use would require data provenance, privacy, representativeness, fairness, and legal review.
- Inference must load the saved estimator and preprocessing objects; it must not refit encoders on incoming data.
