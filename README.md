This project is a real-time face recognition-based attendance system built using Python, OpenCV, and the face_recognition library. The system uses a webcam to detect and recognize students’ faces and automatically logs their attendance into a CSV file with a timestamp.

It is designed for educational institutions, training centers, or any scenario where attendance tracking is required without manual effort.

🔍 Features
✅ Real-time face detection using webcam

✅ Face recognition using deep learning (HOG/CNN)

✅ Automatic attendance marking with timestamp

✅ CSV file export with date-wise logs

✅ Scalable to multiple students

✅ Simple and lightweight with offline capability

🧠 Technologies Used
Python 3.9

OpenCV

face_recognition (built on dlib)

NumPy

CSV file handling

DateTime for timestamping

📂 Project Structure

face_recognition_attendance/

├── faces/                   # Folder with student face images (e.g., ashwin.jpeg)

├── main.py                 # Main Python script

├── README.md               # Project description

└── attendance_logs/        # Generated CSV files with logs (optional)

🛠 How It Works
The system loads known student images and encodes their faces.

It activates the webcam and detects faces in real time.

If a face matches a known encoding, attendance is recorded with the current time.

The system prevents duplicate entries in the same session.

Output is saved in a .csv file named with the current date.


 

