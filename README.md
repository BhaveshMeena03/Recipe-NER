# Recipe NER — Key Entity Extraction

A **Named Entity Recognition** model built with **Conditional Random Fields (CRF)** that extracts key entities from free-text recipe ingredient lists, classifying each token as an **ingredient**, **quantity**, or **unit**.

Structuring messy ingredient text this way enables downstream features for recipe-management systems, dietary-tracking apps, and grocery e-commerce.

## Example

```
input:  "6 Karela Bitter Gourd ... 1 Onion 3 tablespoon Gram flour besan ..."
labels: quantity=6, ingredient="Karela Bitter Gourd", quantity=1, ingredient="Onion",
        quantity=3, unit="tablespoon", ingredient="Gram flour besan" ...
```

## Approach

- Parse the JSON dataset of ingredient strings with token-level NER labels
- Engineer per-token features (word shape, neighbouring tokens, is-numeric, suffixes)
- Train a linear-chain CRF (`sklearn-crfsuite`) to label the token sequence
- Evaluate with entity-level precision / recall / F1 and persist the trained model

## Tech

Python, sklearn-crfsuite, pandas, scikit-learn (Jupyter Notebook)

## Files

| File | Description |
|---|---|
| `Identifying_Key_Entities_in_Recipe_Data_Bhavesh_Meena.ipynb.ipynb` | Feature engineering, training and evaluation |
| `ingredient_and_quantity.json` | Labelled recipe NER dataset |
| `crf_model.pkl` | Trained CRF model |
| `Identifying_Key_Entities_in_Recipe_Data_Bhavesh_Meena.pdf` | Written report |
