# Composer

Este repositorio contiene la implementación del código utilizado para la construcción, entrenamiento y entonación de redes neuronales RNN-LSTM dedicadas a la composición de música automática.

## Estructura del repositorio

Para asegurar la correcta ejecución del programa se recomienda utilizar las carpetas provistas de la siguiente manera:

- midi: Contiene el conjunto de archivos en formado MIDI (.mid) de las melodías a ser utilizadas.
- models: Contiene los modelos de redes generados a través del programa.
- dict: Contiene los diccionarios que mapean notas a enteros y viceversa. Cada vez que se realiza alguna modificación al conjunto de melodías es necesario generar nuevos diccionarios.
- notes: 

## Ejecución y sub-comandos

A continuación se explica como funciona la ejecución del programa a través de cada subcomando.

### Train

Permite construir y entrenar modelos de redes RNN y RNN-LSTM. Por ejemplo:

	python composer.py train --dataset=nottingham

Crearía y luego entrenaría una red neuronal de tipo RNN-LSTM utilizando los valores por defecto provistos en el programa. Si se desea manipular manualmente los valores de los hiperparámetros, es posible a través de argumentos opcionales. Por ejemplo:

	python composer.py train --dataset=nottingham --num_layers=3 --units=512 --learning_rate=0.01 --batch_size=64 --seq_len=16 --epochs=50

Crearía y luego entrenaria una red neuronal de tipo RNN-LSTM de 3 capas LSTM, utilizando 512 unidades por capa, una tasa de aprendizaje de 0.01, lotes de entrenamiento de tamaño 64 y secuencias de tamaño 16 para un interválo de 50 épocas.

### Generate

Permite la generación de música utilizando modelos existentes previamente entrenados. Por ejemplo, suponiendo la existencia de un modelo ya entrenado para una red RNN-LSTM de 3 capas LSTM, 512 unidades por capa LSTM y secuencias de tamaño 100, en la forma de un archivo de pesos denominado *weights-nottingham-512-0.001-0.3-64-50-0.05.hdf5*, podemos generar música de la siguiente manera:

	python composer.py generate --dataset=nottingham --num_layers=3 --units=512 --seq_len=100

Por defecto se genera una composición de 500 notas de longitud, este valor puede manipularse a través de un argumento opcional:

	python composer.py generate --dataset=nottingham --num_layers=3 --units=512 --seq_len=100 --composition_len=50

Generaría alternativamente una composición de 50 notas de longitud.

### Randomize

### Markov

### Display

Produce una partitura musical en formato PDF (.pdf) de un archivo MIDI (.mid). Solo se quiere introducir la ubicación del archivo MIDI a convertir:

	pyton composer.py display --midi_path=melodia.mid
