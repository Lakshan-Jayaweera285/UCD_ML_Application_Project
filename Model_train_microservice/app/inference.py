from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.datasets import cifar10
from PIL import Image
import random
import io

def load_and_predict_image(model_path='cifar10_model.h5'):
    model = load_model(model_path)

    class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
   
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    random_number = random.randint(0, len(X_test) - 1)
    img_array = X_test[random_number:random_number + 1]  
    img_array = img_array.astype('float32') / 255.0  
    true_label = class_labels[y_test[random_number][0]]
    print(f"Using random CIFAR-10 test image with true label: {true_label}")

    #prediction
    prediction = model.predict(img_array)
    
    predicted_class = int(np.argmax(prediction, axis=1)[0])
    prediction_probs = prediction[0].tolist()

    return {
        "predicted_class": class_labels[predicted_class],
        "class_index": predicted_class,
        "probabilities": prediction_probs,
        "true_label": true_label 
    }
