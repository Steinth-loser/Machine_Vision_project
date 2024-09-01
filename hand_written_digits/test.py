import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model           
                                                         
model = load_model('trained_model.h5')                   

img_path = 'three.jpg'  # Path to the image you want to test
img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Normalize the image

# Display the image
plt.imshow(img, cmap='gray')
plt.show()

# Make a prediction
predictions = model.predict(img_array)

# Interpret the prediction
predicted_class = np.argmax(predictions, axis=-1)
print(f"Predicted class: {predicted_class[0]}")