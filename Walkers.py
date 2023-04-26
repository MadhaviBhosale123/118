import cv2
from cv2 import _OUTPUT_ARRAY_DEPTH_MASK_16F


# Create our body classifier
body_classifier = cv2.CascadeClassifier('F:\Madhavi Folder\PRO-106-ProjectTemplate-main\haarcascade_fullbody.xml')


# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    
    
    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray,1.2,3)
    
    # Extract bounding boxes for any bodies identified
    for(x,y,w,h) in bodies:
        cv2.rectangle(bodies, (100,100), (200,500), (0, 255, 0), -1)
    cv2.imshow('output',frame) 
    print(bodies)
    
    

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
