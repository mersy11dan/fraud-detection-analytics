# Class Imbalance Handling Summary

## Technique
- **Method:** `smote` on the training set only
- **Rationale:** SMOTE synthesizes minority-class examples in feature space while keeping all legitimate training transactions.
- **Test set:** never resampled (stratified holdout)

## Split sizes
- Training rows before resampling: **120,889**
- Training rows after resampling: **219,136**
- Test rows (unchanged): **30,223**

## Class distribution comparison

### Training set — before resampling
| class | count | pct |
| --- | --- | --- |
| 0 | 109568 | 90.6352 |
| 1 | 11321 | 9.3648 |

### Training set — after resampling
| class | count | pct |
| --- | --- | --- |
| 0 | 109568 | 50.0 |
| 1 | 109568 | 50.0 |

### Test set — holdout (unmodified)
| class | count | pct |
| --- | --- | --- |
| 0 | 27393 | 90.6363 |
| 1 | 2830 | 9.3637 |

## Interim takeaway
Resampling is applied only to training data so evaluation on the holdout set still reflects natural fraud prevalence. SMOTE improves the model's exposure to rare fraud patterns without discarding legitimate transactions.