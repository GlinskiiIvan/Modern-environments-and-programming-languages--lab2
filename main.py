import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

# Загрузка данных
dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:60].astype(float)
Y = dataset[:, 60]

# Кодирование меток
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)


# Функция для создания и обучения модели
def build_and_train_model(layer_sizes, input_dim, epochs=100, batch_size=10, validation_split=0.1):
    model = Sequential()
    model.add(Dense(layer_sizes[0], input_dim=input_dim, kernel_initializer="normal", activation="relu"))

    # Добавляем скрытые слои в зависимости от их количества
    for size in layer_sizes[1:]:
        model.add(Dense(size, kernel_initializer="normal", activation="relu"))

    # Добавляем выходной слой
    model.add(Dense(1, kernel_initializer="normal", activation="sigmoid"))

    # Компиляция модели
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Обучение модели
    history = model.fit(X, encoded_Y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    return history


# Функция для построения графиков
def create_train_charts(filename, history):
    data = []
    # Данные точности
    data.append({
        'label': 'Точность при обучении',
        'title': 'Точность',
        'val_label': 'Точность при валидации',
        'val_values': history.history.get('val_accuracy'),
        'values': history.history['accuracy'],
    })
    # Данные потерь
    data.append({
        'label': 'Потери при обучении',
        'title': 'Потери',
        'val_label': 'Потери при валидации',
        'val_values': history.history.get('val_loss'),
        'values': history.history['loss'],
    })

    # Определение количества эпох
    epochs = range(1, len(data[0]['values']) + 1)

    # Создание графиков
    figure, axes = plt.subplots(len(data), 1, figsize=(7, 10))
    plt.subplots_adjust(hspace=.4)

    for i, axis in enumerate(axes):
        axis.grid(visible=True, color='lightgray', which='both', zorder=0)
        axis.plot(epochs, data[i]['values'], '.-', label=data[i]['label'], color='g', zorder=3)
        if data[i]['val_values']:
            axis.plot(epochs, data[i]['val_values'], label=data[i]['val_label'], color='r', zorder=3)
        axis.set_title(data[i]['title'])
        axis.set_xlabel('Эпохи')
        axis.set_ylabel(data[i]['title'])
        axis.legend()

    # Сохранение графиков
    figure.savefig(filename)
    plt.close()


# Проведение экспериментов

# 1. Влияние количества нейронов (1 слой, разные нейроны)
histories_neurons = []
neuron_counts = [15, 30, 45]
for count in neuron_counts:
    history = build_and_train_model([count], input_dim=60)
    create_train_charts(f"train_neurons_{count}.png", history)
    histories_neurons.append((count, history))

# 2. Влияние количества слоев (фиксированное число нейронов)
histories_layers = []
layer_configs = [[30], [30, 15], [30, 15, 10]]
for config in layer_configs:
    history = build_and_train_model(config, input_dim=60)
    create_train_charts(f"train_layers_{len(config)}.png", history)
    histories_layers.append((config, history))

# Сравнение моделей по результатам истории обучения
def compare_histories(histories, title_prefix):
    """Функция для сравнения результатов нескольких моделей."""
    # Инициализация графиков для точности и потерь
    plt.figure(figsize=(14, 6))

    # Сравнение точности обучения
    plt.subplot(1, 2, 1)
    for config, history in histories:
        plt.plot(history.history['accuracy'], label=f"Слои: {config}")
    plt.title(f'{title_prefix} - Точность при обучении')
    plt.xlabel('Эпохи')
    plt.ylabel('Точность')
    plt.legend()

    # Сравнение потерь при обучении
    plt.subplot(1, 2, 2)
    for config, history in histories:
        plt.plot(history.history['loss'], label=f"Слои: {config}")
    plt.title(f'{title_prefix} - Потери при обучении')
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.legend()

    plt.show()

# 1. Сравнение влияния количества нейронов на одном слое
compare_histories(histories_neurons, "Влияние количества нейронов")

# 2. Сравнение влияния количества слоев
compare_histories(histories_layers, "Влияние количества слоев")