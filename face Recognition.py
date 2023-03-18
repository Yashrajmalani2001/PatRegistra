

import cv2;
import numpy as np;
import face_recognition;

# Load the known images and encode their face
img_elon = face_recognition.load_image_file('elon.jpg')
elon_encoding = face_recognition.face_encodings(img_elon)[0]

img_bill = face_recognition.load_image_file('bill.jpg')
bill_encoding = face_recognition.face_encodings(img_bill)[0]

# Create a list of known face encodings and their names
known_face_encodings = [
    elon_encoding,
    bill_encoding
]

known_face_names = [
    "Elon Musk",
    "Bill Gates"
]

# Start capturing the video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Resize the frame to speed up face recognition
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR to RGB
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and their encodings in the frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Loop through each face in this frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        # Check if the face is a match for the known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        # If a match was found, display the name
        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]
        else:
            name = "Unknown"

        # Draw a rectangle around the face
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Exit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
