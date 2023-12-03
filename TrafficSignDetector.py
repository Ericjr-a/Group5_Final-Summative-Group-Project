import streamlit as st
import cv2
import numpy as np
import pickle
from matplotlib import pyplot as plt
import time



model_file = open("TrafficSignDetectorModel.p" , "rb")
model = pickle.load(model_file)

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
    
def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/225
    return img

def getClassName(classNo):
    if   classNo == 0: return 'Speed Limit 20 km/h'
    elif classNo == 1: return 'Speed Limit 30 km/h'
    elif classNo == 2: return 'Speed Limit 50 km/h'
    elif classNo == 3: return 'Speed Limit 60 km/h'
    elif classNo == 4: return 'Speed Limit 70 km/h'
    elif classNo == 5: return 'Speed Limit 80 km/h'
    elif classNo == 6: return 'End of Speed Limit 80 km/h'
    elif classNo == 7: return 'Speed Limit 100 km/h'
    elif classNo == 8: return 'Speed Limit 120 km/h'
    elif classNo == 9: return 'No passing'
    elif classNo == 10: return 'No passing for vechiles over 3.5 metric tons'
    elif classNo == 11: return 'Right-of-way at the next intersection'
    elif classNo == 12: return 'Priority road'
    elif classNo == 13: return 'Yield'
    elif classNo == 14: return 'Stop'
    elif classNo == 15: return 'No vechiles'
    elif classNo == 16: return 'Vechiles over 3.5 metric tons prohibited'
    elif classNo == 17: return 'No entry'
    elif classNo == 18: return 'General caution'
    elif classNo == 19: return 'Dangerous curve to the left'
    elif classNo == 20: return 'Dangerous curve to the right'
    elif classNo == 21: return 'Double curve'
    elif classNo == 22: return 'Bumpy road'
    elif classNo == 23: return 'Slippery road'
    elif classNo == 24: return 'Road narrows on the right'
    elif classNo == 25: return 'Road work'
    elif classNo == 26: return 'Traffic signals'
    elif classNo == 27: return 'Pedestrians'
    elif classNo == 28: return 'Children crossing'
    elif classNo == 29: return 'Bicycles crossing'
    elif classNo == 30: return 'Beware of ice/snow'
    elif classNo == 31: return 'Wild animals crossing'
    elif classNo == 32: return 'End of all speed and passing limits'
    elif classNo == 33: return 'Turn right ahead'
    elif classNo == 34: return 'Turn left ahead'
    elif classNo == 35: return 'Ahead only'
    elif classNo == 36: return 'Go straight or right'
    elif classNo == 37: return 'Go straight or left'
    elif classNo == 38: return 'Keep right'
    elif classNo == 39: return 'Keep left'
    elif classNo == 40: return 'Roundabout mandatory'
    elif classNo == 41: return 'End of no passing'
    elif classNo == 42: return 'End of no passing by vechiles over 3.5 metric tons'

def predicting(camera, threshold):
    # Displaying message
    st.write('Turning Camera On...')
    st.write()

    # Setting up parameters
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Stopping Prediction Button
    stop_button = st.button('Stop Predicting')

    # Create a placeholder for displaying the video feed
    video_placeholder = st.empty()

    while True:
        # Reading image from camera
        success, imgOrignal = camera.read()

        # Resizing image
        img = np.asarray(imgOrignal)
        img = cv2.resize(img, (32, 32))

        # Preprocessing image
        img = preprocessing(img)

        # Reshaping image
        img = img.reshape(1, 32, 32, 1)

        # Writing class and probability on original image
        cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

        # Predicting image
        predictions = model.predict(img)
        classIndex = np.argmax(predictions)
        probabilityValue = np.amax(predictions)

        video_placeholder.image(imgOrignal, channels="BGR", use_column_width=True)
        # Checking prediction accuracy
        if probabilityValue > threshold:
            # Writing class and probability on original image
            cv2.putText(imgOrignal, str(classIndex) + " " + str(getClassName(classIndex)), (120, 35), font, 0.75,
                        (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2,
                        cv2.LINE_AA)

            # Displaying prediction on streamlit
            if probabilityValue > 0.998 and classIndex != 13:  # removed class: 13 yield since the model wrongfully detects it
                st.text(f"Predicted Class: {classIndex} - {getClassName(classIndex)}")
                st.text(f"Probability: {round(probabilityValue * 100, 2)}%")

            # Displaying original image with predictions
            video_placeholder.image(imgOrignal, channels="BGR", use_column_width=True)

        # Checking stop button status
        if stop_button:
            camera.release()
            st.stop()

def main_page():
    st.title('Traffic Sign Recognition AI')
    st.subheader('By Eric Afari and Sedem Amediku')
    st.write("\nModel Accuracy on Direct Sign Image Input: 99.7%")
    st.write("\nModel Accuracy on Camera Input: 72%")
    st.write('Click the start button and point your camera towards a traffic sign to start detecting..')
    start_button = st.button('Start Predicting')
    threshold = 0.9  # Probability Threshold

    if start_button:
        # Capturing Video
        camera = cv2.VideoCapture(0)
        predicting(camera, threshold)

if __name__ == "__main__":
    main_page()