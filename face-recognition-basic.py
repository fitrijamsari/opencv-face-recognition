import cv2
import numpy as np
import face_recognition

# load image convert image BGR2RGB
image_elon = face_recognition.load_image_file("imageBasic/elon_musk.jpg")
image_elon = cv2.cvtColor(image_elon, cv2.COLOR_BGR2RGB)

# load test image convert image BGR2RGB
img_test = face_recognition.load_image_file("imageBasic/elon_musk_test.jpg")
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

# find face location in our image
face_location = face_recognition.face_locations(image_elon)[0]
# print(face_location)
# (206, 883, 527, 562) => top, right, bottom, left line

# using encoder to identify faces in image
elon_encoding = face_recognition.face_encodings(image_elon)[0]
# draw rectangle
cv2.rectangle(
    image_elon,
    (face_location[3], face_location[0]),
    (face_location[1], face_location[2]),
    (255, 0, 255),
    3,
)

face_location_test = face_recognition.face_locations(img_test)[0]
test_encoding = face_recognition.face_encodings(img_test)[0]
cv2.rectangle(
    img_test,
    (face_location_test[3], face_location_test[0]),
    (face_location_test[1], face_location_test[2]),
    (255, 0, 255),
    3,
)

# compare both faces and find the distance between them using linear SEM
res = face_recognition.compare_faces([elon_encoding], test_encoding)

# if too many dataset later, there might be some similarity of difference faces. to reduce this effects we find the distance. The lower the distance the better match.
faceDis = face_recognition.face_distance([elon_encoding], test_encoding)

# boolean result, TRUE meaining image and test image are the same person. False for different person
print(res, faceDis)

cv2.putText(
    img_test,
    f"{res} {round(faceDis[0],2)}",
    (50, 50),
    cv2.FONT_HERSHEY_COMPLEX,
    1,
    (0, 0, 255),
    2,
)

cv2.imshow("Elon Musk", image_elon)
cv2.imshow("Elon Musk Test", img_test)
cv2.waitKey(0)