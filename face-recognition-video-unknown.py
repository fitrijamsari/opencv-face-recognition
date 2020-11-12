import cv2
import numpy as np
import face_recognition
import os

path = "imageAttendance"
images = []
classNames = []
myList = os.listdir(path)
# print(myList)

# use filename and append on images [] and classNames []
for cls in myList:
    currentImg = cv2.imread(f"{path}/{cls}")
    images.append(currentImg)
    # append only name of file eg bill_gates not bill_gates.jpg
    classNames.append(os.path.splitext(cls)[0])
# print(classNames)

# create function for list of encoded images
def findEncodings(images):
    # create empty list for the encoding
    encodeList = []
    # loop through all images, then encode image, then append on the encodelist
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)
# print(len(encodeListKnown))
print("Encoding Complete")

# initialize webcam to feed images/video and compare with the encoder

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    # reduce size / scale down the image to 1/4 of original size
    img_resize = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)

    # find location of ALL faces in image frame
    facesCurFrame = face_recognition.face_locations(img_resize)
    encodeCurFrame = face_recognition.face_encodings(img_resize, facesCurFrame)

    # itterate all faces in current frame, then compare with the encoding of known face
    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        # find matching
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        # count face distance
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        # find the lowest value of faceDis as our match face with the encodeListKnown (real face)
        matchIndex = np.argmin(faceDis)

        # display bounding box and write name
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            # multiple by 4 back since we resize the original image to 1/4 previously
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )
            cv2.rectangle(
                img,
                (x1, y2 - 20),
                (x2, y2),
                (0, 255, 0),
                cv2.FILLED,
            )
            cv2.putText(
                img,
                name,
                (x1 + 4, y2 - 4),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        else:
            y1, x2, y2, x1 = faceLoc
            # multiple by 4 back since we resize the original image to 1/4 previously
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2,
            )
            cv2.rectangle(
                img,
                (x1, y2 - 20),
                (x2, y2),
                (255, 0, 0),
                cv2.FILLED,
            )
            cv2.putText(
                img,
                "unknown",
                (x1 + 6, y2 - 6),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    cv2.imshow("Webcam", img)
    # cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


# improvement: add create encoder based on multiple same person images for better result
