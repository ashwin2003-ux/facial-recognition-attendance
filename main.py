import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Open webcam
video_capture = cv2.VideoCapture(0)

# Load known faces
ashwin_image = face_recognition.load_image_file("faces/ashwin.jpeg")
ashwin_encoding = face_recognition.face_encodings(ashwin_image)[0]

loki_image = face_recognition.load_image_file("faces/loki.jpeg")
loki_encoding = face_recognition.face_encodings(loki_image)[0]

arjun_image = face_recognition.load_image_file("faces/arjun.jpeg")
arjun_encoding = face_recognition.face_encodings(arjun_image)[0]

rekha_image = face_recognition.load_image_file("faces/rekha.jpeg")
rekha_encoding = face_recognition.face_encodings(rekha_image)[0]

mona_image = face_recognition.load_image_file("faces/mona.jpeg")
mona_encoding = face_recognition.face_encodings(mona_image)[0]

lokesh_image = face_recognition.load_image_file("faces/lokesh.jpeg")
lokesh_encoding = face_recognition.face_encodings(lokesh_image)[0]

known_face_encodings = [ashwin_encoding, loki_encoding, arjun_encoding,mona_encoding,rekha_encoding,lokesh_encoding]
known_face_names = ["ashwin", "loki","arjun","rekha","mona","lokesh"]

# Copy names for attendance tracking
attendance_list = known_face_names.copy()

# Prepare CSV file
now = datetime.now()
current_date = now.strftime("%d-%m-%y")
csv_filename = f"{current_date}.csv"
with open(csv_filename, "w+", newline="") as f:
    lnwriter = csv.writer(f)
    lnwriter.writerow(["Name", "Time"])

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Resize frame for faster recognition
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Face detection & encoding
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                now_time = datetime.now().strftime("%H:%M:%S")

                if name in attendance_list:
                    attendance_list.remove(name)
                    lnwriter.writerow([name, now_time])
                    print(f"Attendance marked for {name} at {now_time}")

                # Display name on screen
                font = cv2.FONT_HERSHEY_TRIPLEX
                bottom_left = (10, 100)
                font_scale = 1.5
                font_color = (255, 0, 0)
                thickness = 1
                line_type = 2
                cv2.putText(frame, name + " present", bottom_left, font, font_scale, font_color, thickness, line_type)

        # Show frame
        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
