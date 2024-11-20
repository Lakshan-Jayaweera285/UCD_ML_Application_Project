from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from app.db import save_training_details
from keras.regularizers import l2

def train_cifar10_model(epochs: int, batch_size: int, validation_split: float, learning_rate: float):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Normalize pixels
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0


    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001), input_shape=(32, 32, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu' , kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu' , kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax') 
    ])

    learning_rate = float(learning_rate)  
    optimizer = Adam(learning_rate=learning_rate)

    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    test_loss, test_acc = model.evaluate(X_test, y_test)
    val_loss = history.history['val_loss'][-1]
    val_acc = history.history['val_accuracy'][-1]

    test_loss, test_acc = model.evaluate(X_test, y_test)

    save_training_details(epochs, batch_size, learning_rate, validation_split, test_acc, test_loss, val_acc, val_loss)

    model.save("cifar10_model.h5")
    print("Model training complete and saved as cifar10_model.h5")
    print(f"Test accuracy: {test_acc}")
    print(f"Test loss: {test_loss}")
    print(f"Validation accuracy: {val_acc}")
    print(f"Validation loss: {val_loss}")

    training_metrics = {
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "validation_accuracy": val_acc,
        "validation_loss": val_loss
    }


    return training_metrics
