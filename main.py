import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def create_model(values, problem_number, epochs=1000, learning_rate=0.01, activation='sigmoid'):
    # Create a neural network model
    model = tf.keras.Sequential()
    for layer in values[0]:
        if activation == 'sigmoid':
            model.add(tf.keras.layers.Dense(units=len(layer), activation='sigmoid'))
        elif activation == 'relu':
            model.add(tf.keras.layers.Dense(units=len(layer), activation='relu')

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    history = model.fit(inputs, outputs, epochs=epochs, verbose=0)
    
    # Plot decision boundary and show accuracy
    plot_decision_boundary(model, values, inputs, outputs, problem_number, activation)
    print(f'Problem {problem_number} - Activation: {activation} - Accuracy: {history.history["accuracy"][-1]:.2f} - Loss: {history.history["loss"][-1]:.4f}')

def plot_decision_boundary(model, values, inputs, outputs, problem_number, activation):
    h = 0.01
    x_min, x_max = inputs[:, 0].min() - 1, inputs[:, 0].max() + 1
    y_min, y_max = inputs[:, 1].min() - 1, inputs[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = model.predict(grid)
    predictions = np.round(predictions).astype(int)
    predictions = predictions.reshape(xx.shape)
    
    plt.contourf(xx, yy, predictions, cmap='viridis', alpha=0.8)
    plt.scatter(inputs[:, 0], inputs[:, 1], c=outputs, cmap='cool', edgecolor='k')
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title(f'Problem {problem_number} - Activation: {activation} Decision Boundary')
    plt.show()

# Define input data and outputs
inputs = np.array([(0, 0), (0, 1), (1, 0), (1, 1)])
outputs = np.array([0, 1, 1, 0])

# Problem 1
values_p1 = [np.array([[1, 1], [1, 1, -2]]), np.array([-1.5, -0.5])]

# Problem 2
values_p2 = [np.array([[1, 1], [1, 1, 1], [1, -1, -1]]), np.array([-0.5, -1.5, -0.5])]

# Problem 3
values_p3 = [np.array([[-2, 9.2], [4.3, 8.8, -4.5], [5.3, 0, 0]]), np.array([-1.8, -0.1, -0.8])]

# Solve Problem 1 with different hyperparameters
create_model(values_p1, 1, epochs=2000, learning_rate=0.001, activation='sigmoid')
create_model(values_p1, 1, epochs=2000, learning_rate=0.01, activation='relu')

# Solve Problem 2 with different hyperparameters
create_model(values_p2, 2, epochs=2000, learning_rate=0.001, activation='sigmoid')
create_model(values_p2, 2, epochs=2000, learning_rate=0.01, activation='relu')

# Solve Problem 3 with different hyperparameters
create_model(values_p3, 3, epochs=2000, learning_rate=0.001, activation='sigmoid')
create_model(values_p3, 3, epochs=2000, learning_rate=0.01, activation='relu')
