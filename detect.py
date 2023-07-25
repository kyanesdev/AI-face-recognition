import cv2
import face_recognition
import numpy as np
import time

# Step 1: Load the contents of the 'perfil.txt' file
known_face_encodings = np.loadtxt("perfil.txt")

# Step 2: Reshape the array to have two dimensions
known_face_encodings = np.reshape(known_face_encodings, (1, -1))

# Step 3: Calculate the norm along axis 1
norm = np.linalg.norm(known_face_encodings, axis=1)
known_face_names = ["Usuario identificado"]  # Add the names corresponding to the known face encodings

# Initialize the video capture using the default camera
cap = cv2.VideoCapture(0)

delay = 3  # Delay in seconds
start_time = time.monotonic()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)

    # Perform face detection
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    elapsed_time = time.monotonic() - start_time

    for face_location, face_encoding in zip(face_locations, face_encodings):
            # Compare the face encoding with the known face encodings
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Usuario desconocido"
            
    if elapsed_time < delay:
        # Display gray rectangle and "Identificando..." text
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (128, 128, 128), 2)
        cv2.putText(frame, "Identificando...", (left, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
    else:
            if True in matches:
                # Find the index of the matched face encoding
                matched_index = matches.index(True)
                name = known_face_names[matched_index]
                # Draw rectangle and display the name
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            else:
                # Draw rectangle and display the name
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()