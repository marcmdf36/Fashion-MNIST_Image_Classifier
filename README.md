# Fashion-MNIST Image Classifier

CNN de clasificación de imágenes entrenada sobre el dataset [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), que contiene 70 000 imágenes en escala de grises (28×28 px) de 10 categorías de ropa.

Desarrollado como práctica final de un microcredencial en IA.

## Stack

Python · TensorFlow/Keras · scikit-learn · Matplotlib · NumPy

## Arquitectura

Red convolucional de 3 bloques seguida de una cabeza clasificadora:

```
Input (28×28×1)
  → Conv2D(32) + BN + ReLU + MaxPool  → 14×14×32
  → Conv2D(64) + BN + ReLU + MaxPool  → 7×7×64
  → Conv2D(128) + BN + ReLU + GlobalAvgPool → 128
  → Dense(256) + BN + ReLU + Dropout(0.4)
  → Dense(10) + Softmax
```

Decisiones clave: BatchNormalization para estabilizar el entrenamiento, GlobalAveragePooling en lugar de Flatten para reducir parámetros y regularizar implícitamente, L2 en capas densas donde se concentra el riesgo de sobreajuste.

## Entrenamiento

- Split estratificado: 48k train / 12k validación / 10k test (test nunca visto hasta evaluación final)
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
- Reproducibilidad garantizada mediante semilla global fija (SEED = 42)

## Uso

```bash
pip install tensorflow scikit-learn matplotlib
jupyter notebook MarcMorera_PracticaFinal.ipynb
```
