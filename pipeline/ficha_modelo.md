# Ficha técnica del modelo

Edite los valores después de `:`. Este archivo sirve como base de los campos descriptivos para los próximos entrenamientos. Los campos técnicos y los valores de métricas se obtienen directamente del entrenamiento.

## Campos editables
- cliente: Colcan
- empresa: Colcan
- responsable_tecnico: Edward Arroyave Agudelo
- product_owner: William Alfonso Vigoya
- scrum_master: Viviana León Parra
- objetivo: Entrenar un modelo de detección de casetes para distinguir los de muestra simple de los de muestra múltiple e identificar resultados positivos o negativos.
- alcance: El modelo analizará imágenes de casetes y distinguirá el tipo de muestra y el resultado detectado. Su desempeño se evaluará con los conjuntos de validación y prueba disponibles.
- mejoras: Facilitar la identificación automática del tipo de casete y de su resultado, y generar los pesos del modelo para su integración en el proyecto.
- origen_datos: Imágenes compartidas por Colcan mediante un endpoint y almacenadas en Blob Storage.
- pruebas: Evaluación sobre los conjuntos de validación y prueba disponibles para medir la calidad de la detección y revisar los resultados por clase.
- plan_reentrenamiento: Incorporar un flujo automatizado con revisión humana de las nuevas imágenes y sus anotaciones antes de utilizarlas para reentrenar el modelo.
- criterios_reentrenamiento: Revisar mensualmente la disponibilidad de nuevas imágenes y ejecutar el reentrenamiento cuando se cuente con al menos 50 imágenes nuevas validadas.
- responsable_monitoreo: Edward Alexander Arroyave Agudelo

## Observaciones de métricas
- map_50_95: Precisión media de detección promediada entre umbrales IoU de 0.50 a 0.95; resume la calidad general.
- ap50: Precisión media de detección con IoU mínimo de 0.50; tolera ubicaciones menos exactas.
- ap75: Precisión media de detección con IoU mínimo de 0.75; exige ubicaciones más exactas.
- precision: Proporción de detecciones positivas que fueron correctas.
- recall: Proporción de objetos reales que el modelo detectó.
- average_recall: Recuperación promedio de objetos a distintos umbrales IoU.
- inference_ms: Tiempo promedio de inferencia por imagen, en milisegundos; menor es mejor.
