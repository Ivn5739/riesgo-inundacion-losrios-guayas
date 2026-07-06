# Proyecto Final – Clasificación del Riesgo de Inundación por Parroquia

Este README documenta las secciones correspondientes al **Tema 4** del proyecto: implementación de SVM, construcción de un modelo ensamble, ajuste de hiperparámetros con GridSearchCV y evaluación mediante curvas ROC/AUC.

## Contenido

- `AA_Proyecto_FinalPRUEBA2.ipynb` — notebook completo del proyecto.
- `informe.tex` / `informe.pdf` — informe breve en LaTeX de las secciones descritas abajo.

## Tema 4

## 1. Implementación SVM

Se entrena un `SVC` con kernel RBF (`kernel='rbf'`), `probability=True` (necesario para calcular la curva ROC/AUC más adelante) y `C=1.0`. Al ser sensible a la escala de las variables, se entrena sobre los datos ya estandarizados (`X_train_scaled`) con el mismo `StandardScaler` usado para la Regresión Logística.

**Resultado:** F1-Score ponderado (test) = **0.8056**.

## 2. Construcción del modelo ensamble (RL + DT + SVM + RF)

Se agrega un **Árbol de Decisión** (`max_depth=8`) como componente adicional obligatorio del ensamble, y se combina junto con Regresión Logística, SVM y Random Forest mediante un `VotingClassifier` con **voto suave** (`voting='soft'`), que promedia las probabilidades de cada modelo en lugar de solo contar votos por mayoría.

**Resultados (test):**
| Modelo | F1 ponderado |
|---|---|
| Árbol de Decisión | 0.8601 |
| Ensamble (RL+DT+SVM+RF) | 0.8601 |

## 3. GridSearchCV en ≥1 modelo (aplicado a SVM)

Se optimiza el SVM mediante `GridSearchCV` con validación cruzada de 5 particiones, explorando:
- `C`: [0.1, 1, 10]
- `kernel`: ['rbf', 'linear']
- `gamma`: ['scale', 'auto']

**Mejores hiperparámetros:** `{'C': 10, 'gamma': 'scale', 'kernel': 'linear'}`
**Mejor F1 ponderado (CV):** 0.9505
**F1 ponderado en test (SVM optimizado):** 0.9045

## 4. Curva ROC, AUC – comparación de modelos

Dado que el problema es multiclase (bajo / medio / alto riesgo), se binarizan las etiquetas (estrategia One-vs-Rest) y se calcula la curva ROC y el AUC por clase para los 6 modelos entrenados, promediando de forma macro. Ver `roc_curve.png`.

**Comparación de AUC (macro), ordenada de mayor a menor:**

| Modelo | AUC macro | AUC micro |
|---|---|---|
| SVM optimizado (GridSearch) | 0.9966 | 0.9943 |
| Ensamble (RL+DT+SVM+RF) | 0.9728 | 0.9728 |
| SVM | 0.9592 | 0.9739 |
| Random Forest | 0.9524 | 0.9490 |
| Regresión Logística | 0.9320 | 0.9467 |
| Árbol de Decisión | 0.8929 | 0.8929 |

**Conclusión:** el SVM optimizado por GridSearchCV obtiene el mejor desempeño global (AUC macro más alto), superando incluso al modelo ensamble, lo que sugiere que el ajuste fino de hiperparámetros aportó más valor que la combinación de modelos en este caso.
