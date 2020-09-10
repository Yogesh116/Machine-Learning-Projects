import cv2
import numpy as np
face_classifier=cv2.CascadeClassifier('C:/Users/yogesh/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

#will have to extract the face feature

def face_extractor(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # i convert the RGB image into gray scale because it is easy to perform operation

    #Now we must call our classifier function,
    # passing it some very important parameters, as scale factor, number of neighbors and minimum size of the detected face.

    faces=face_classifier.detectMultiScale(gray,1.3,5)

    if faces is(): #if any face is not here. So return none
        return None


    for(x,y,w,h) in faces: #if face is here
        cropped_faces=img[y:y+h,x:x+w]

    return cropped_faces

cap=cv2.VideoCapture(0)  #i set the camera
count = 0 #initialization for counting photos

while True:
    ret,frame=cap.read() #load our input video in grayscale mode
    if face_extractor(frame) is not None: #face is detect
        count+=1
        face=cv2.resize(face_extractor(frame),(200,200)) #camera frame resize
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)  #convert into gray scale

        file_paths='C:/Users/yogesh/Downloads/CSE contents/faces/user'+str(count)+'.jpg' #To save the photo and saved by series like 1,2,3,etc and format jpg
        cv2.imwrite(file_paths,face)

        cv2.putText(face,str(count),(100,100),cv2.FONT_HERSHEY_COMPLEX,1,2) #put the number counting in front of camera
        cv2.imshow('Face Cropper', face)
    else:
        print("Face Not Found")
        pass

    if cv2.waitKey(1) == 13 or count==200:
        break

cap.release()
cv2.destroyAllWindows()
print('Colleting Samples Complete!!!')






