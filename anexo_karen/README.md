# Anexo: análisis complementario (Karen Manríquez)

Versión extendida del análisis de churn. **No es la solución principal del proyecto**: el informe oficial es el [`README.md`](../README.md) de la raíz y su notebook [`notebooks/churn_telco_ML.ipynb`](../notebooks/churn_telco_ML.ipynb).

Este anexo compara tres modelos (Regresión Logística, Random Forest y HistGradientBoosting), usa PR-AUC como métrica de selección e incluye pruebas extra (SMOTE, curvas de aprendizaje, SHAP). Por eso sus resultados no son idénticos a los del informe principal.

## Estructura

```
anexo_karen/
├── notebooks/telco_churn.ipynb   ← notebook del anexo
├── src/
│   ├── preprocessing.py          ← limpieza, variables derivadas y ColumnTransformer
│   └── modeling.py               ← pipelines y búsqueda de hiperparámetros
└── requirements.txt              ← incluye imbalanced-learn y shap
```

El notebook lee los datos desde `../data/Telco_Customer_Churn_Dataset.csv` (la carpeta `data/` de la raíz del repo), así que no hay que copiar el CSV.

## Cómo ejecutarlo

```bash
pip install -r anexo_karen/requirements.txt
cd anexo_karen/notebooks
jupyter notebook telco_churn.ipynb   # Kernel → Restart & Run All
```

Al ejecutarlo se crean las carpetas `images/`, `models/`, `reports/` y `data/processed/` dentro de `anexo_karen/`.
