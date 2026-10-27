# Predicción de abandono de clientes (Telco Customer Churn)

**MLY1101 – Machine Learning · Evaluación Parcial N°2 · Caso A**
**Integrantes:** Nicolás Garrido · Karen Manríquez · Felipe Espinoza
**Metodología:** CRISP-DM · **Lenguaje:** Python 3 (pandas, scikit-learn)

> Informe técnico reproducible del proyecto. Todo el código está en [`notebooks/churn_telco.ipynb`](notebooks/churn_telco.ipynb) y los gráficos en [`images/`](images/).

---

## Resumen ejecutivo

- La compañía pierde al **26,5 %** de sus clientes; esos clientes representan **USD 139.131 de ingreso mensual (30,5 % del total)**.
- Se entrenaron y compararon dos modelos supervisados. El **Random Forest** (modelo final) detecta al **76 % de los clientes que abandonan** con **ROC-AUC = 0,846** en datos no vistos, y el 20 % de clientes con mayor score concentra **2,5 veces** más churn que el promedio.
- Una segmentación **K-Means (k = 4)** revela que el segmento **“Fibra mes a mes”** (31 % de la cartera) tiene **54 % de churn** y concentra **≈ 72 % del ingreso en riesgo**.
- Una campaña dirigida por el modelo captura el **78 % del ingreso en riesgo** contactando solo al **37 %** de los clientes y genera un beneficio neto estimado **44 % mayor** que contactar a todos.
- Principales impulsores del churn: **contrato mes a mes, baja antigüedad, fibra óptica, falta de soporte técnico/seguridad online y pago con cheque electrónico.**

---

## 1. Descripción del problema de negocio

Una empresa de telecomunicaciones registra una alta tasa de abandono de clientes (*churn*). Captar un cliente nuevo cuesta considerablemente más que retener uno existente, y hoy las acciones de retención son reactivas (se actúa cuando el cliente ya pidió la baja) o masivas (se ofrece lo mismo a todos, con alto costo).

**Pregunta de negocio:** ¿qué clientes tienen mayor probabilidad de abandonar en el próximo período, qué los caracteriza y cómo priorizar las acciones de retención?

**Formulación como problema de ML:**
- **Clasificación binaria supervisada:** predecir `Churn` (Yes/No) a partir del perfil, servicios y condiciones de cuenta del cliente.
- **Aprendizaje no supervisado (clustering):** segmentar la cartera para diseñar acciones diferenciadas por grupo.

## 2. Objetivos del proyecto

**Objetivo general:** desarrollar un modelo de Machine Learning que identifique a los clientes con alta probabilidad de abandono, para focalizar las campañas de retención y reducir la pérdida de ingresos.

**Objetivos específicos:**
1. Analizar los datos para identificar patrones, problemas de calidad y variables asociadas al churn (EDA).
2. Preparar y transformar los datos de forma reproducible y sin fuga de información.
3. Implementar y comparar **dos modelos supervisados** (Regresión Logística y Random Forest) con validación cruzada y ajuste de hiperparámetros.
4. Aplicar **K-Means** para descubrir segmentos de clientes y relacionarlos con el riesgo de abandono.
5. Traducir las métricas en impacto económico y recomendaciones de negocio, evaluando además el impacto ético del modelo.

## 3. Definición de KPIs

| Tipo | KPI | Por qué | Meta | Resultado (test) |
|---|---|---|---|---|
| Modelo | **Recall** de la clase churn | Un churner no detectado es ingreso perdido; es el error más caro | ≥ 0,75 | **0,759 ✓** |
| Modelo | **ROC-AUC** | Capacidad de ordenar clientes por riesgo, independiente del umbral | ≥ 0,80 | **0,846 ✓** |
| Modelo | **Lift top 20 %** | Cuánto mejor que el azar es contactar al 20 % de mayor riesgo | ≥ 2,0 | **2,49 ✓** |
| Negocio | **Tasa de churn** | Indicador principal que el proyecto busca reducir | Línea base 26,5 % → reducir | Línea base medida |
| Negocio | **% del ingreso mensual en riesgo capturado** | Mide si el modelo apunta a los clientes que más valen | ≥ 70 % | **78 % ✓** |
| Negocio | **Beneficio neto de la campaña** | Ingreso retenido − costo de contactar | > “contactar a todos” | **+44 % ✓** |
| Ética | **Diferencia de recall por género** | Equidad del modelo | < 5 p.p. | 5,9 p.p., no significativa (p = 0,18) ⚠ |

## 4. Descripción de las fuentes de datos

| Atributo | Detalle |
|---|---|
| Fuente | *Telco Customer Churn* — IBM Sample Data Sets (publicado en Kaggle), entregado por la asignatura |
| Archivo | [`data/Telco_Customer_Churn_Dataset.csv`](data/Telco_Customer_Churn_Dataset.csv) |
| Tamaño | 7.043 filas (clientes) × 21 columnas |
| Granularidad | 1 fila = 1 cliente; `Churn` indica si abandonó en el último mes |

| Grupo | Variables |
|---|---|
| Identificador | `customerID` |
| Demográficas | `gender`, `SeniorCitizen` (0/1), `Partner`, `Dependents` |
| Cuenta | `tenure` (meses), `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Servicios | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Objetivo | `Churn` (Yes / No) |

## 5. Metodología utilizada (CRISP-DM)

```
 Comprensión del negocio → Comprensión de los datos → Preparación → Modelado → Evaluación → Despliegue
          ↑_________________________________________________________________________|
```

| Fase | Qué se hizo | Dónde |
|---|---|---|
| 1. Comprensión del negocio | Problema, objetivos, KPIs y supuestos económicos | README §1–3 · Notebook §1 |
| 2. Comprensión de los datos | Calidad de datos, distribución del objetivo, análisis univariado y bivariado, correlaciones | README §6 · Notebook §2 |
| 3. Preparación de los datos | Limpieza, recodificación, selección de variables, One-Hot, escalado, partición estratificada, `Pipeline` | README §7 · Notebook §3 |
| 4. Modelado | Regresión Logística y Random Forest con `GridSearchCV` (5 folds); K-Means para segmentación | README §8–9 · Notebook §4–5 |
| 5. Evaluación | Métricas técnicas en test, umbral, interpretabilidad, impacto económico, auditoría ética | README §10–11 · Notebook §6 |
| 6. Despliegue | Modelo y segmentador serializados (`joblib`), scoring de clientes priorizado | README §12 · Notebook §7 |

El proceso fue **iterativo**: p. ej., tras ver en el modelado que `TotalCharges` distorsionaba los coeficientes de la regresión logística se volvió a la fase de preparación para excluirla.

## 6. Análisis exploratorio de los datos (EDA)

### 6.1 Calidad de datos

| Revisión | Resultado | Tratamiento |
|---|---|---|
| Nulos explícitos | 0 | — |
| Duplicados (filas / `customerID`) | 0 / 0 | — |
| `TotalCharges` como texto | 11 registros con espacio en blanco, **todos con `tenure = 0`** (clientes nuevos aún no facturados) | Convertir a numérico e imputar **0** (no es un faltante aleatorio) |
| Codificación inconsistente | `SeniorCitizen` 0/1 vs Yes/No en el resto | Recodificar a Yes/No |
| Categorías redundantes | “No internet service”, “No phone service” | Recodificar a “No” |
| Outliers | Rangos de negocio válidos (tenure 0–72, cargos 18–119 USD) | Se conservan |

### 6.2 Variable objetivo
El dataset está **desbalanceado: 73,5 % No / 26,5 % Yes**. Por eso no se usa *accuracy* como métrica principal (un modelo que siempre dice “No” obtiene 73,5 %), se estratifica la partición y se usa `class_weight="balanced"`.

![Distribución churn](images/01_distribucion_churn.png)

### 6.3 Patrones identificados

![Numéricas vs churn](images/02_numericas_vs_churn.png)

- **Antigüedad:** mediana de 10 meses en quienes abandonan vs 38 en quienes se quedan. El riesgo se concentra en el primer año.
- **Cargo mensual:** quienes abandonan pagan más (mediana USD 80 vs 64).

![Tasa de churn por variable categórica](images/04_tasa_churn_categoricas.png)

| Variable | Categoría de mayor riesgo | Churn | Comparación |
|---|---|---|---|
| `Contract` | Month-to-month | 43 % | 11 % (1 año), 3 % (2 años) |
| `InternetService` | Fiber optic | 42 % | 19 % DSL, 7 % sin internet |
| `OnlineSecurity` / `TechSupport` | No | 42 % | 15 % con el servicio |
| `PaymentMethod` | Electronic check | 45 % | 15–19 % otros métodos |
| `SeniorCitizen` | Sí | 42 % | 24 % |
| `Partner` / `Dependents` | No | 33 % / 31 % | 20 % / 15 % |
| `gender`, `PhoneService` | — | ≈ igual | diferencia < 2 p.p. → **no relevantes** |

![Interacciones](images/05_interacciones.png)

- El churn cae con la antigüedad dentro de cada tipo de contrato; el peor caso es **mes a mes con menos de 6 meses**.
- La fibra sin soporte técnico es la combinación más riesgosa.
- `TotalCharges` tiene correlación 0,83 con `tenure` (**multicolinealidad**).

### 6.4 Variables relevantes (ranking por diferencia de tasa de churn entre categorías)
`Contract` (40 p.p.) > `InternetService` (35) > `OnlineSecurity` (34) > `TechSupport` (34) > `OnlineBackup` (33) > `DeviceProtection` (32) > `PaymentMethod` (30) > … > `gender` (0,8).

## 7. Preparación y transformación de los datos

| Paso | Decisión | Justificación |
|---|---|---|
| Limpieza | `TotalCharges` → numérico, blancos = 0 | Clientes con `tenure = 0` |
| Exclusión | `customerID` | Identificador, sin valor predictivo; dato personal |
| Exclusión ética | `gender` | Atributo protegido sin poder predictivo (EDA); se usa solo para auditar sesgo |
| Exclusión técnica | `TotalCharges` | ≈ `tenure × MonthlyCharges`; multicolinealidad. Excluirla no reduce el ROC-AUC en CV (0,846 → 0,845) y mejora la interpretabilidad |
| Recodificación | “No internet/phone service” → “No”; `SeniorCitizen` → Yes/No | Evitar columnas redundantes tras One-Hot |
| Ingeniería de variables | `num_servicios`, `contrato_ord`, `fibra`, `pago_auto` | Variables resumen para la segmentación K-Means |
| Codificación | One-Hot Encoding (`drop="if_binary"`, `handle_unknown="ignore"`) → 24 variables | Requerido por scikit-learn; robusto a categorías nuevas |
| Escalado | `StandardScaler` en `tenure` y `MonthlyCharges` | Necesario para Regresión Logística y K-Means |
| Partición | 80/20 **estratificada** (`random_state=42`): 5.634 train / 1.409 test | Mantiene 26,5 % de churn en ambos |
| Anti-fuga | Todo dentro de un `Pipeline` + `ColumnTransformer` | El preprocesamiento se ajusta solo con train (y dentro de cada fold) |

## 8. Modelos supervisados

| Modelo | Justificación de la selección | Hiperparámetros explorados | Mejor configuración |
|---|---|---|---|
| Baseline (`DummyClassifier`) | Referencia mínima | — | Clase mayoritaria |
| **Regresión Logística** | Modelo lineal interpretable (odds ratio), estándar en churn, probabilidades bien calibradas | `C` ∈ {0,01…5}, penalización L1/L2 | `C=5`, L2, `class_weight="balanced"` |
| **Random Forest** | Captura no linealidades e interacciones (contrato × antigüedad), robusto a multicolinealidad, entrega importancia de variables | `max_depth`, `min_samples_leaf`, `max_features` (300 árboles) | `max_depth=None`, `min_samples_leaf=20`, `max_features=0.3`, `class_weight="balanced"` |

- **Validación:** `GridSearchCV` con `StratifiedKFold` (5 folds), métrica ROC-AUC.
- **Umbral de decisión:** elegido con predicciones *out-of-fold* en train (sin tocar test) maximizando F1 sujeto a recall ≥ 0,75 → 0,557 (RL) y 0,541 (RF).

### Resultados en test (1.409 clientes no vistos)

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Lift top 20 % |
|---|---|---|---|---|---|---|---|
| Baseline (Dummy) | 0,735 | 0,000 | 0,000 | 0,000 | 0,500 | 0,265 | 1,05 |
| Regresión Logística | 0,752 | 0,524 | 0,733 | 0,611 | 0,839 | 0,629 | 2,51 |
| **Random Forest** | **0,765** | **0,541** | **0,759** | **0,632** | **0,846** | **0,651** | 2,49 |

ROC-AUC en validación cruzada: RL 0,845 ± 0,012 · RF **0,848 ± 0,010**.

![ROC, PR y comparación](images/07_roc_pr_comparacion.png)
![Matrices de confusión](images/08_matrices_confusion.png)

**Modelo final: Random Forest.** Tiene el mejor ROC-AUC en CV (criterio fijado antes de ver el test), es el único que cumple el KPI de recall y obtiene mejor F1 y PR-AUC. Como la diferencia con la Regresión Logística es pequeña, ésta se mantiene como **modelo explicativo** para comunicar la dirección de cada factor.

![Importancia de variables](images/09_importancia_variables.png)

Ambos modelos coinciden: **contrato mes a mes, antigüedad, contrato de 2 años (protector), fibra óptica, cargo mensual y pago con cheque electrónico** son las variables más influyentes. En la Regresión Logística, tener fibra multiplica las odds de abandono por ≈ 3,9 y el contrato de 2 años las reduce a ≈ 0,32.

## 9. Modelo no supervisado: segmentación K-Means

- **Variables (estandarizadas):** `tenure`, `MonthlyCharges`, `num_servicios`, `contrato_ord`, `fibra`, `pago_auto`. **No se usa `Churn`** para formar los grupos.
- **Elección de k:** método del codo + silueta (k = 2…8). Se eligió **k = 4** (codo entre 3 y 4, silueta 0,33 > k = 2 y 3, segmentos accionables).

![Elección de k](images/10_eleccion_k.png)
![Segmentos](images/11_segmentos_kmeans.png)
![Centroides](images/12_centroides_kmeans.png)

| Segmento | % cartera | Perfil | Churn | % ingreso en riesgo | Acción propuesta |
|---|---|---|---|---|---|
| **S0 Fibra mes a mes** | 31 % | ~21 meses, USD 87/mes, 100 % fibra, 94 % mensual | **54 %** | **72 %** | Prioridad 1: migración a contrato anual con beneficio, revisar precio/calidad de fibra, pago automático |
| S1 Básicos nuevos mes a mes | 29 % | ~14 meses, USD 40/mes, DSL o solo teléfono | 25 % | 15 % | Onboarding primer año, venta cruzada de seguridad/soporte |
| S3 Premium leales | 24 % | ~60 meses, USD 91/mes, 5 servicios, contrato largo | 10 % | 12 % | Fidelización y renovación anticipada |
| S2 Básicos con contrato | 17 % | ~47 meses, USD 30/mes, contrato 1–2 años | 3 % | 1 % | Sin inversión en retención; upselling |

## 10. Interpretación del desempeño y traducción al negocio

**Lectura de las métricas del modelo final:**
- **Recall 0,76:** de cada 100 clientes que realmente abandonan, el modelo detecta 76.
- **Precisión 0,54:** de cada 100 clientes señalados, 54 efectivamente abandonan (al azar serían 26,5).
- **ROC-AUC 0,85:** si se toma un cliente que abandona y uno que no, el modelo asigna mayor riesgo al primero el 85 % de las veces.
- **Lift 2,5:** contactar al 20 % de mayor riesgo es 2,5 veces más efectivo que contactar al azar.

**Simulación económica** (supuestos: USD 50 por contacto, 35 % de éxito de retención, valor = 12 meses de `MonthlyCharges`):

| Escenario (test, 1.409 clientes) | Contactados | Churners detectados | % ingreso en riesgo capturado | Beneficio neto (USD) |
|---|---|---|---|---|
| No hacer nada | 0 | 0 | 0 % | 0 |
| Contactar a todos | 1.409 | 374 | 100 % | 43.853 |
| Regresión Logística | 523 | 274 | 77 % | 62.330 |
| **Random Forest** | **525** | **284** | **78 %** | **63.255** |

![Beneficio vs umbral](images/13_beneficio_vs_umbral.png)

El umbral es una **decisión de negocio**: con estos costos, bajarlo a ≈ 0,34 maximizaría el beneficio (≈ USD 68.500) a cambio de contactar más clientes. Escalado a toda la cartera, el potencial es del orden de USD 300.000 por ciclo de campaña.

## 11. Evaluación del impacto ético

| Aspecto | Medida adoptada |
|---|---|
| Privacidad | Se elimina `customerID`; no hay datos de contacto. Uso alineado con la normativa chilena de datos personales (Ley 19.628 / Ley 21.719) |
| Atributos protegidos | `gender` excluido del modelo; usado solo para auditoría |
| Equidad (género) | Recall 0,73 (F) vs 0,79 (M): 5,9 p.p., **no significativa** (test z, p = 0,18). FPR similar (0,22 vs 0,24). Se recomienda monitorear |
| Equidad (adultos mayores) | Mayor recall (0,85) y FPR (0,42) por su mayor churn real (44 % vs 23 %). La acción es ofrecer un beneficio, por lo que no los perjudica; cuidar frecuencia y canal de contacto |
| Uso responsable | El score se usa solo para retención (nunca para subir precios o negar servicios), con revisión humana y reentrenamiento periódico |

## 12. Conclusiones y recomendaciones

1. El Random Forest cumple los KPIs técnicos y de negocio, y supera claramente a la línea base y a la estrategia de contactar a todos.
2. **Migrar clientes mes a mes a contratos anuales**, priorizando el segmento S0 (fibra mes a mes).
3. **Revisar la oferta de fibra óptica** (precio/calidad).
4. **Incluir seguridad online y soporte técnico** en paquetes de retención.
5. **Incentivar el pago automático** y **reforzar el onboarding** del primer año.
6. Medir la campaña con un **grupo de control** y reentrenar trimestralmente.

**Limitaciones:** una sola fotografía temporal, sin datos de uso ni reclamos; supuestos económicos estimados; pocos casos para auditar subgrupos. **Próximos pasos:** Gradient Boosting, calibración de probabilidades, variables de comportamiento.

---

## Estructura del proyecto

```
telco_churn_ml/
├── README.md                     ← este informe técnico
├── requirements.txt              ← dependencias
├── data/
│   ├── Telco_Customer_Churn_Dataset.csv   ← datos originales
│   └── scoring_clientes_test.csv          ← salida: clientes de test priorizados por riesgo
├── notebooks/
│   └── churn_telco.ipynb         ← notebook ejecutado, documentado por fases CRISP-DM
├── models/
│   ├── modelo_churn_final.joblib ← Pipeline (preprocesamiento + Random Forest)
│   ├── kmeans_segmentacion.joblib← scaler + K-Means + nombres de segmentos
│   ├── config_modelo.json        ← umbral, variables y métricas
│   └── metricas_test.csv         ← tabla comparativa de modelos
├── images/                       ← 13 gráficos generados por el notebook
└── presentacion/
    └── guion_presentacion.md     ← guion de 10 min y preguntas de defensa
```

## Cómo reproducir

```bash
# 1. Crear entorno e instalar dependencias
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Ejecutar el notebook (desde la carpeta notebooks/)
cd notebooks
jupyter notebook churn_telco.ipynb   # Kernel → Restart & Run All
```

- Las rutas son relativas (`../data`, `../models`, `../images`) y la semilla es fija (`RANDOM_STATE = 42`): los resultados son reproducibles.
- El notebook vuelve a generar todos los gráficos de `images/` y los archivos de `models/` (≈ 2 min).
- Los `.joblib` se crearon con scikit-learn 1.9; si se usa otra versión, basta con re-ejecutar el notebook para regenerarlos.

**Uso del modelo guardado:**
```python
import joblib, json, pandas as pd
modelo = joblib.load("models/modelo_churn_final.joblib")
umbral = json.load(open("models/config_modelo.json"))["umbral"]
# clientes_nuevos: DataFrame con las columnas originales del CSV, transformado con la
# función ingenieria_variables() del notebook (§3) para aplicar las mismas recodificaciones
prob = modelo.predict_proba(clientes_nuevos)[:, 1]
contactar = prob >= umbral
```
