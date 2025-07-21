import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np

image_path = r'C:\Number_Prediction\static\images\sample1.png'
img = cv2.imread(image_path)

reader = easyocr.Reader(['en'], gpu=False)

text_ = reader.readtext(img)

threshold = 0.25  #it is the threshold for the score of the text detection
# You can adjust this value based on your requirements 

for t_, t in enumerate(text_):
    print(t)

    bbox, text, score = t

    if score > threshold:
        cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 3)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()