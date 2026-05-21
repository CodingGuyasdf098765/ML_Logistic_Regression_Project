# Clasificador de Regresión Logística — Campaña de Marketing Bancario

> Pipeline de clasificación binaria sobre más de 41.000 registros de clientes bancarios: EDA completo con ingeniería de características, selección chi-cuadrado y un modelo de Regresión Logística optimizado con GridSearchCV en 1.400 combinaciones de modelos — alcanzando una precisión del 92,2%.

---

## Problema

Un banco portugués quiere identificar qué clientes existentes tienen más probabilidad de suscribirse a un depósito a largo plazo tras una campaña de marketing telefónico. Concentrar el alcance en los clientes con mayor probabilidad reduce las llamadas desperdiciadas y mejora el ROI de la campaña. Este es un problema de clasificación binaria: ¿se suscribió el cliente (`yes`) o no (`no`)?

## Dataset

- **Fuente:** Dataset Bank Marketing Campaign (UCI / 4Geeks)
- **Tamaño:** 41.188 filas × 21 columnas → 41.176 tras eliminar 12 duplicados
- **Target:** `y` — se suscribió al depósito a largo plazo (yes/no)
- **Balance de clases:** ~89% no / ~11% yes — fuertemente desbalanceado

**Tipos de columnas:**

| Tipo | Columnas |
|---|---|
| Numéricas | age, duration, campaign, pdays, previous, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed |
| Categóricas | job, marital, education, default, housing, loan, contact, month, day_of_week, poutcome, y |

## Pipeline de EDA y Preprocesamiento

| Paso | Acción |
|---|---|
| Duplicados | 12 eliminados → 41.176 filas |
| Ingeniería de características | `pdays=999` (valor centinela "nunca contactado") → columna binaria `was_previously_contacted` |
| Columnas eliminadas | `default` (varianza casi nula, ~99% "no") y `pdays` (reemplazada por la característica ingenierizada) |
| Eliminación de outliers | `duration` limitado a 644,5s; `campaign` limitado a 6 llamadas → 35.951 filas |
| Escalado | MinMaxScaler en todas las columnas de características |
| Selección de características | SelectKBest (chi², k=7) — aplicado solo al conjunto de entrenamiento para evitar fuga de datos |

**Características seleccionadas:** `month`, `was_previously_contacted`, `previous`, `poutcome`, `emp.var.rate`, `euribor3m`, `nr.employed`

**Hallazgo clave del EDA:** `duration` (duración de la llamada) es el predictor individual más fuerte de suscripción — pero constituye una fuga de datos ya que solo se conoce después de que termina la llamada. Eliminada del conjunto final de características para un modelo realista en producción.

## Resultados del Modelo

| Modelo | Precisión |
|---|---|
| Línea base (hiperparámetros por defecto) | **92,07%** |
| Optimizado (GridSearchCV) | **92,16%** |

**Espacio de búsqueda de GridSearchCV:** C (7 valores) × penalty (4 opciones) × solver (5 opciones) = 140 combinaciones × validación cruzada de 10 pliegues = **1.400 modelos entrenados**

**Mejores hiperparámetros:** `C=0.1`, `penalty=l2`, `solver=liblinear`

La mejora de precisión es modesta (+0,08%) porque el modelo de línea base con la configuración por defecto ya estaba bien ajustado para este dataset. El valor del grid search aquí es la confirmación fundamentada, no un salto dramático.

## Conclusiones Clave

- **La ingeniería de características supera a la selección de características:** Convertir el centinela `pdays=999` en un indicador binario significativo captura información real que los números crudos ocultan.
- **La fuga de datos es sutil:** `duration` parece una característica legítima, pero conocer la duración de la llamada antes de realizarla es imposible — incluirla inflaría la precisión en entrenamiento pero produciría un modelo que no puede desplegarse.
- **Alta precisión ≠ buen modelo cuando las clases están desbalanceadas:** El 92% suena sólido, pero un modelo naive que prediga "no" para cada cliente alcanzaría ~89%. La precisión y el recall en la clase minoritaria "yes" cuentan la historia real.

## Stack Tecnológico

`Python` · `scikit-learn` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn`

## Ejecutar Localmente

```bash
git clone https://github.com/matthewkane-ml/ML_LogisticRegression_MTK.git
cd ML_LogisticRegression_MTK
pip install -r requirements.txt
jupyter notebook src/app.ipynb
```

El modelo entrenado se guarda en `models/` mediante `pickle`.

## Próximos Pasos

- Evaluar con **precisión, recall y F1** para la clase minoritaria en lugar de la precisión general — el coste empresarial de perder un posible suscriptor es mucho mayor que una llamada desperdiciada
- Probar `class_weight="balanced"` o sobremuestreo **SMOTE** para mejorar el recall en la clase "yes"
- Construir una **curva de calibración** para comprobar si las probabilidades predichas del modelo (no solo las predicciones binarias) son suficientemente fiables para usarse como puntuación de ranking de clientes

---

**Autor:** Matthew Kane — [LinkedIn](https://www.linkedin.com/in/thomas-k-392094410/) · [Portafolio GitHub](https://github.com/matthewkane-ml)
