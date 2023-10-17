# Rendimiento-RNNs
Estudio del rendimiento de RNNs y su comportamiento al variar arquitecturas e hiperparámetros.

## Estudiantes/Autores
- Agustín González
- Diego Torreblanca

## Descripción

Análisis y estudio de redes neuronales capaces de reconocer las palabras dichas en audios de voz, haciendo uso de Redes Neuronales Recurrentes (RNN en inglés). Se prueban 3 tipos de RNN, multi-layer Elman RNN, Long Short Term Memory RNN (LSTM) y multi-layer gated recurrent unit RNN (GRU). 

Para lo anterior se implementan múltiples modelos que son entrenados y validados sobre un subconjunto del dataset [SPEECHCOMMANDS](https://arxiv.org/abs/1804.03209), y se grafican curvas de evolución de la _loss_ y _accuracy_, además de matrices de confusión y otras métricas. Todo el código se encuentra dentro del jupyter notebook [Proyecto_RNN_10a](/Proyecto_RNN_10a.ipynb).

El subconjunto del dataset está dado por test_list.txt, val_list.txt y train_list.txt

## Modo de uso

### Dependencies

* Python 3.10+
  Librerías externas requeridas:
  * torch
  * IPython
  * matplotlib
  * sklearn
  * seaborn
  * tqdm

#### Ejecución

Al trabajar con jupyter notebooks, se deben ir ejecutando los bloques siguiendo el orden de aparición, los bloques que deben ejecutarse para que el resto de código pueda correr son los que se encuentran en las secciones: "Importar librerías y archivos", "Carga de los datasets", "Gráficos" y "Funciones de entrenamiento". El resto de bloques debe ejecutarlos según los experimentos que desee probar. Los experimentos son independientes entre sí, por lo que puede correrlos en el orden que desee. 

## Historial de versiones

* 0.1
  * Entrega Preliminar

## License

This project is licensed under the MIT License - see the LICENSE.md file for details
