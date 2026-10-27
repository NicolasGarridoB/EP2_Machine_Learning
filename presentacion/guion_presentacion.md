# Guion de presentación — EP2 Machine Learning (10 minutos)

**Caso A: Telco Customer Churn** · Nicolás · Karen · Felipe
Mostrar el `README.md` (o el notebook) y apoyarse en los gráficos de `images/`.

> La defensa es **individual con preguntas cruzadas**: cada integrante debe dominar TODO el proyecto, no solo su parte. Revisar el banco de preguntas al final.

| Bloque | Tiempo | Quién | Indicador de la pauta |
|---|---|---|---|
| 1. Problema, KPIs y CRISP-DM | 2:00 | Integrante 1 | IE5 Metodología |
| 2. EDA y preparación de datos | 2:30 | Integrante 1 | IE5 Metodología |
| 3. Modelos supervisados | 2:30 | Integrante 2 | IE6 Supervisado |
| 4. K-Means (no supervisado) | 1:30 | Integrante 3 | IE7 No supervisado |
| 5. Resultados de negocio, ética y cierre | 1:30 | Integrante 3 | IE8 Interpretación |

---

## 1. Problema, KPIs y metodología (2:00)

> “Una empresa de telecomunicaciones pierde al **26,5 %** de sus clientes, lo que equivale a **USD 139 mil de ingreso mensual, el 30 % del total**. Retener es más barato que captar, pero hoy las campañas son masivas o llegan tarde. Nuestro objetivo es **predecir qué clientes van a abandonar** para focalizar la retención.
> Lo planteamos como **clasificación binaria** con dos modelos supervisados, más una **segmentación K-Means** para diseñar acciones por grupo.
> Definimos KPIs técnicos —**recall ≥ 0,75**, porque no detectar a un cliente que se va es el error más caro; **ROC-AUC ≥ 0,80**; **lift ≥ 2**— y KPIs de negocio: **capturar ≥ 70 % del ingreso en riesgo** y **ganar más que contactar a todos**. También un KPI ético de equidad por género.
> Seguimos **CRISP-DM**, y fue iterativo: por ejemplo, volvimos a preparación para sacar `TotalCharges` al ver que distorsionaba la regresión logística.”

## 2. EDA y preparación (2:30) — gráficos 01, 04, 05

> “El dataset tiene **7.043 clientes y 21 variables**, sin nulos ni duplicados. El único problema de calidad: **11 valores en blanco en `TotalCharges`**, todos con antigüedad 0 → clientes nuevos sin facturar, así que imputamos **0**, no la media.
> La variable objetivo está **desbalanceada 73/27**, por eso **no usamos accuracy** como métrica principal.
> Patrones: el **contrato mes a mes** tiene 43 % de churn contra 3 % en dos años; la **fibra óptica** 42 %; **sin soporte técnico o seguridad** 42 %; **cheque electrónico** 45 %. Los que se van tienen una mediana de **10 meses** de antigüedad. **Género no influye** (menos de 1 punto).
> Preparación: eliminamos `customerID`; sacamos **`gender`** por ética y porque no aporta; sacamos **`TotalCharges`** por multicolinealidad (0,83 con tenure) sin perder AUC; One-Hot a las categóricas, StandardScaler a las numéricas, split **80/20 estratificado**, y todo dentro de un **Pipeline** para evitar fuga de datos.”

## 3. Modelos supervisados (2:30) — gráficos 07, 08, 09

> “Comparamos **Regresión Logística**, por ser interpretable con odds ratio, y **Random Forest**, porque captura interacciones como contrato × antigüedad. Ambos con `class_weight='balanced'` por el desbalance, y ajustamos hiperparámetros con **GridSearchCV, 5 folds estratificados, optimizando ROC-AUC**.
> El umbral no lo dejamos en 0,5: lo elegimos con predicciones out-of-fold en train para maximizar F1 con recall ≥ 0,75, **sin tocar el test**.
> En test: Random Forest **AUC 0,846, recall 0,76, F1 0,63**; Logística AUC 0,839, recall 0,73. Ambos muy sobre el baseline de AUC 0,5.
> Elegimos **Random Forest** porque gana en validación cruzada y es el único que cumple el KPI de recall. Usamos la logística para explicar: la **fibra multiplica por 3,9 las odds** de abandono y el **contrato de 2 años las reduce a un tercio**.”

## 4. K-Means (1:30) — gráficos 10, 11, 12

> “Para el no supervisado usamos **K-Means** sobre seis variables estandarizadas: antigüedad, cargo, número de servicios, tipo de contrato, fibra y pago automático, **sin usar la etiqueta churn**.
> Elegimos **k = 4** con el método del codo y la silueta, priorizando segmentos accionables.
> El hallazgo clave: el segmento **‘Fibra mes a mes’**, 31 % de la cartera, tiene **54 % de churn y concentra el 72 % del ingreso en riesgo**. En el otro extremo, los ‘Básicos con contrato’ tienen 3 %, ahí no conviene gastar en retención.”

## 5. Negocio, ética y cierre (1:30) — gráfico 13

> “¿Qué significa en plata? Con supuestos de USD 50 por contacto y 35 % de éxito, el modelo contacta solo al **37 %** de los clientes y captura el **78 % del ingreso en riesgo**, con un beneficio **44 % mayor** que contactar a todos y 63 % menos de costo.
> En ética: sacamos género y lo auditamos; la diferencia de recall es 5,9 puntos pero **no es estadísticamente significativa** (p = 0,18), así que recomendamos monitorearla. El modelo se usa **solo para ofrecer beneficios**, nunca para subir precios.
> Recomendaciones: **migrar clientes mes a mes a contratos anuales, revisar la fibra, incluir soporte técnico en paquetes y fomentar el pago automático**. Gracias.”

---

## Banco de preguntas de defensa (preparar TODOS)

**Metodología / datos**
1. *¿Por qué imputaron `TotalCharges` con 0 y no con la media?* → Los 11 casos tienen `tenure = 0`: aún no han sido facturados. La media inventaría un cobro inexistente.
2. *¿Qué es la fuga de datos y cómo la evitaron?* → Usar información del test al entrenar. Todo el preprocesamiento está en un `Pipeline` que se ajusta solo con train, también dentro de cada fold de CV; el umbral se eligió con train.
3. *¿Por qué estratificar?* → Para mantener el 26,5 % de churn en train y test y que la evaluación sea representativa.
4. *¿Por qué sacaron `TotalCharges`?* → Es ≈ tenure × MonthlyCharges (corr 0,83 con tenure), genera multicolinealidad que vuelve inestables los coeficientes. Sin ella el AUC en CV pasa de 0,846 a 0,845: no se pierde nada.
5. *¿Por qué One-Hot y no Label Encoding?* → Las categorías no tienen orden; Label Encoding inventaría una distancia entre ellas.
6. *Fases de CRISP-DM.* → Negocio, datos, preparación, modelado, evaluación, despliegue; es cíclico.

**Supervisado**
7. *¿Por qué no accuracy?* → Por el desbalance: decir siempre “No” da 73,5 % de accuracy y 0 de recall (lo muestra el Dummy).
8. *¿Qué significa recall 0,76 y precisión 0,54?* → Detectamos 76 de cada 100 que se van; de cada 100 alertas, 54 son churn real.
9. *¿Qué es ROC-AUC?* → Probabilidad de que el modelo asigne más riesgo a un churner que a un no churner elegido al azar. 0,5 = azar, 1 = perfecto.
10. *¿Qué hace `class_weight='balanced'`?* → Pondera más los errores en la clase minoritaria (churn) en proporción inversa a su frecuencia.
11. *¿Por qué ese umbral (0,54) y no 0,5?* → Se eligió con predicciones out-of-fold para cumplir recall ≥ 0,75 maximizando F1. Además el negocio podría bajarlo a 0,34 para maximizar beneficio.
12. *¿Cómo funciona Random Forest?* → Muchos árboles entrenados con muestras bootstrap y subconjuntos de variables; la predicción es el promedio/votación. Reduce la varianza de un árbol solo.
13. *¿Qué controlan `min_samples_leaf` y `max_features`?* → Tamaño mínimo de hoja (evita sobreajuste) y fracción de variables candidatas por división (diversidad entre árboles).
14. *¿Hay sobreajuste?* → El AUC de CV (0,848) y de test (0,846) son casi iguales: el modelo generaliza.
15. *¿Por qué `MonthlyCharges` tiene coeficiente negativo en la logística si los que pagan más se van más?* → Es un efecto condicional: el precio ya está explicado por los servicios (fibra, streaming), que tienen coeficientes positivos. A igualdad de servicios, el cargo extra no aumenta el riesgo.
16. *¿Qué es un odds ratio?* → exp(coeficiente). Fibra = 3,9 → multiplica por 3,9 las odds de abandono manteniendo lo demás constante.

**No supervisado**
17. *¿Cómo funciona K-Means?* → Asigna cada punto al centroide más cercano y recalcula centroides hasta converger; minimiza la inercia (suma de distancias cuadradas intra-cluster).
18. *¿Por qué escalar antes de K-Means?* → Usa distancia euclidiana; sin escalar, `MonthlyCharges` (hasta 119) dominaría sobre variables 0/1.
19. *¿Cómo eligieron k?* → Codo de la inercia (entre 3 y 4) + silueta (0,33) + interpretabilidad para el negocio.
20. *¿Qué es la silueta?* → Compara la distancia media al propio cluster vs al cluster vecino; va de −1 a 1. 0,33 indica estructura moderada, razonable con datos de clientes.
21. *¿Usaron Churn en el clustering?* → No. Solo después para perfilar el churn de cada segmento.
22. *¿Para qué sirve el PCA del gráfico?* → Solo para visualizar los 6 ejes en 2D (75 % de la varianza); el clustering se hizo en las 6 dimensiones.

**Negocio / ética**
23. *¿De dónde salen los USD 63 mil?* → Ingreso retenido (churners detectados × 35 % éxito × 12 meses de cargo) menos USD 50 por cada cliente contactado, en el test.
24. *¿Qué pasa si cambian los supuestos?* → Cambia el umbral óptimo, no el ranking del modelo; el gráfico 13 permite ajustarlo.
25. *¿El modelo discrimina?* → Género excluido; diferencia de recall 5,9 p.p. no significativa (p = 0,18). Adultos mayores reciben más ofertas porque realmente abandonan más, y la acción es un beneficio.
26. *¿Cómo lo pondrían en producción?* → Score mensual con el `.joblib`, campaña al grupo sobre el umbral, grupo de control A/B y reentrenamiento trimestral por *data drift*.
