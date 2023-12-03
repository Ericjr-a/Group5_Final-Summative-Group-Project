import os
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import legacy

path = '/Users/ericjr/Desktop/AIFProject/myData'
labelFile = '/Users/ericjr/Desktop/AIFProject/labels.csv'

# defining parameters for training
batch_size_val = 64
epochs_val = 500
imageDimesions = (128, 128, 3) 
testRatio = 0.2
validationRatio = 0.2

print("Done1")

# Calculate total samples
total_samples = sum([len(files) for r, d, files in os.walk(path)])
train_samples = int(total_samples * (1 - testRatio - validationRatio))

# Numerically sort the folders for consistency
def numerical_sort(value):
    return int(value) if value.isdigit() else value

class_folders = sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))], key=numerical_sort)
noOfClasses = len(class_folders)

print("Done2")

# Load and preprocess images
images = []
classNo = []
for class_id, folder_name in enumerate(class_folders):
    folder_path = os.path.join(path, folder_name)
    if not os.path.isdir(folder_path):
        continue
    image_files = os.listdir(folder_path)
    for image_file in image_files:
        img_path = os.path.join(folder_path, image_file)
        curImg = cv2.imread(img_path)
        curImg = cv2.resize(curImg, (imageDimesions[0], imageDimesions[1])) # Resize images for ResNet50
        if curImg is not None:
            images.append(curImg)
            classNo.append(class_id)

images = np.array(images)
classNo = np.array(classNo)
print("Done3")

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(classNo)
print("Done4")

# Splitting dataset into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(images, encoded_labels, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)

print("Done5")


data = pd.read_csv(labelFile)

print("Data shape: ", data.shape, type(data))
print("Done6")





# Number of columns for the subplot (number of sample images per class)
# Number of images to display per class
cols = 5

# Creating a figure with subplots
fig, axs = plt.subplots(nrows=noOfClasses, ncols=cols, figsize=(5 * cols, 4 * noOfClasses))
print("Done7")


# Loop through each class
for class_id in range(noOfClasses):
    # Get the first 'cols' number of images of the current class
    class_images = X_train[y_train == class_id][:cols]

    # Displaying images for each class
    for col in range(cols):
        if col < len(class_images):
            image = class_images[col]
            axs[class_id, col].imshow(image, cmap=plt.get_cmap("gray"))  # Adjust cmap if images are not grayscale
            axs[class_id, col].axis("off")
            if col == 0:  # Label the first image of each row with the class name
                if not data[data['ClassId'] == class_id].empty:
                    class_name = data.loc[data['ClassId'] == class_id, 'SignName'].values[0]
                    axs[class_id, col].set_title(f"{class_id}-{class_name}")
                else:
                    axs[class_id, col].set_title(f"Class ID {class_id}")

plt.tight_layout()
plt.show()
print("Done8")



num_of_samples = []

# Count the number of samples for each class
for class_id in range(noOfClasses):
    num_of_samples.append(np.sum(y_train == class_id))

# plotting the bar chart
plt.figure(figsize=(12, 4))
plt.bar(range(0, noOfClasses), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

num_classes = data['SignName'].nunique()  
print("Number of classes:", num_classes)

print("Done9")



# Preprocess images for ResNet50
def preprocess_images(image_set):
    processed_set = []
    for img in image_set:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB as ResNet50 was trained on RGB images
        img = cv2.resize(img, (imageDimesions[0], imageDimesions[1])) # Resize images for ResNet50
        img = img / 255 # Normalize pixel values
        processed_set.append(img)
    return np.array(processed_set)

X_train = preprocess_images(X_train)
X_validation = preprocess_images(X_validation)
X_test = preprocess_images(X_test)

print("Done10")



# Image augmentation
dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
dataGen.fit(X_train)

print("Done11")

# Displaying augmented image samples
batches = dataGen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)
print("Done12")


fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()
for i in range(15):
    # If your images are grayscale and have been reshaped to (32, 32, 1), use squeeze to remove the last dimension
    image_to_display = X_batch[i].squeeze()
    axs[i].imshow(image_to_display, cmap=plt.get_cmap("gray"))
    axs[i].axis('off')
plt.show()
print("Done13")


# Loading the ResNet50 pre-trained model
resnet_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=imageDimesions))
print("Done14")

# Freezing the layers of the base model
for layer in resnet_model.layers:
    layer.trainable = False

print("Done15")

# Adding custom layers
x = GlobalAveragePooling2D()(resnet_model.output)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(noOfClasses, activation='softmax')(x)
model = Model(inputs=resnet_model.input, outputs=output_layer)
print("Done16")



print("Done18")






# Compiling the model
model.compile(optimizer=legacy.Adam(learning_rate=0.001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

print("Done19")

# Training the model
history = model.fit(
    dataGen.flow(X_train, y_train, batch_size=batch_size_val),
    steps_per_epoch=len(X_train) // batch_size_val,
    epochs=epochs_val,
    validation_data=(X_validation, y_validation)
)
print("Done20")


# Plotting training results
plt.figure(1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
print("Done21")

plt.figure(2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
print("Done22")

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
print("Done23")

# Save the model
pickle_out = open("SignDetectorModel_ResNet50.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()
print("Done24")
