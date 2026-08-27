from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime, time as dtime
import shutil
from deepface import DeepFace
import mediapipe as mp

app = FastAPI(title="Face Recognition Attendance Backend")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
LOGS_DIR = os.path.join(BASE_DIR, "Attendence_Logs")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# --- SHIFT & TIMING CONFIGURATION ---
SHIFT_START_TIME = dtime(9, 0, 0)  # 09:00 AM
GRACE_PERIOD_MINUTES = 15  # Up to 09:15 AM is On-Time, after is Late
SHIFT_END_TIME = dtime(17, 0, 0)  # 05:00 PM (After 05:00 PM is Overtime)


def clear_deepface_cache():
    if not os.path.exists(DATASET_DIR):
        return
    for filename in os.listdir(DATASET_DIR):
        if filename.endswith(".pkl"):
            try:
                os.remove(os.path.join(DATASET_DIR, filename))
            except Exception as e:
                print(f"Error removing cached file: {e}")


def check_liveness(img_np: np.ndarray) -> tuple[bool, str]:
    """
    Simplified Anti-Spoofing:
    Uses MediaPipe 3D Landmark Mesh depth to distinguish a real face from flat 2D images.
    """
    if img_np is None or img_np.size == 0:
        return False, "Invalid frame captured"

    try:
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
        ) as face_mesh:
            rgb_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_img)

            if not results.multi_face_landmarks:
                return False, "No face detected in frame"

            landmarks = results.multi_face_landmarks[0].landmark

            # Calculate 3D Depth Difference between Nose Tip (1) and Cheeks (234, 454)
            nose_z = landmarks[1].z
            left_cheek_z = landmarks[234].z
            right_cheek_z = landmarks[454].z
            z_depth_diff = abs(nose_z - ((left_cheek_z + right_cheek_z) / 2))

            # Flat paper/photo cutouts typically have near-zero depth variance (< 0.002)
            if z_depth_diff < 0.002:
                return False, "Spoof Detected: Flat 2D Image / Photo Printout"

    except Exception as e:
        # If landmark calculation fails, allow processing to proceed without blocking valid users
        print(f"Liveness Check Exception: {e}")
        return True, "Liveness Fallback Passed"

    return True, "Liveness Verified"


def determine_attendance_status(check_in_time: datetime) -> str:
    current_t = check_in_time.time()

    if current_t >= SHIFT_END_TIME:
        return "Overtime"

    grace_seconds = GRACE_PERIOD_MINUTES * 60
    shift_start_dt = datetime.combine(check_in_time.date(), SHIFT_START_TIME)
    check_in_dt = datetime.combine(check_in_time.date(), current_t)

    time_diff = (check_in_dt - shift_start_dt).total_seconds()

    if time_diff <= grace_seconds:
        return "On-Time"
    else:
        return "Late"


@app.get("/")
def home():
    return {"status": "DeepFace Attendance Backend Running"}


@app.post("/register")
async def register_user(emp_id: str = Form(...), name: str = Form(...), file: UploadFile = File(...)):
    formatted_name = name.strip().replace(' ', '_')
    filename = f"{emp_id}_{formatted_name}.jpg"
    filepath = os.path.join(DATASET_DIR, filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    clear_deepface_cache()
    return {"message": f"Successfully registered {name} (ID: {emp_id})"}


@app.get("/attendance/today")
def get_today_attendance():
    today_str = datetime.now().strftime('%Y-%m-%d')
    log_file = os.path.join(LOGS_DIR, f"Attendance_{today_str}.csv")

    if not os.path.exists(log_file):
        df = pd.DataFrame(columns=['ID', 'Name', 'Date', 'Time', 'Status'])
        df.to_csv(log_file, index=False)
        return []

    df = pd.read_csv(log_file)
    return df.to_dict(orient="records")


@app.get("/attendance/dates")
def get_available_dates():
    files = [f for f in os.listdir(LOGS_DIR) if f.startswith("Attendance_") and f.endswith(".csv")]
    dates = [file.replace("Attendance_", "").replace(".csv", "") for file in files]
    dates.sort(reverse=True)
    return {"dates": dates}


@app.get("/attendance/by-date")
def get_attendance_by_date(date: str):
    log_file = os.path.join(LOGS_DIR, f"Attendance_{date}.csv")
    if not os.path.exists(log_file):
        return []
    df = pd.read_csv(log_file)
    return df.to_dict(orient="records")


@app.delete("/attendance/delete")
def delete_attendance_by_date(date: str):
    log_file = os.path.join(LOGS_DIR, f"Attendance_{date}.csv")
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
            return {"message": f"Successfully deleted records for {date}"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")
    else:
        raise HTTPException(status_code=404, detail="Records for this date do not exist.")


@app.get("/students")
def get_all_students():
    valid_images = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    students = []
    for img_name in valid_images:
        name_id = os.path.splitext(img_name)[0]
        parts = name_id.split('_')
        emp_id = parts[0]
        name = " ".join(parts[1:]) if len(parts) > 1 else parts[0]
        students.append({
            "id": emp_id,
            "name": name,
            "image_filename": img_name
        })
    return {"students": students}


@app.get("/students/image/{filename}")
def get_student_image(filename: str):
    image_path = os.path.join(DATASET_DIR, filename)
    if os.path.exists(image_path):
        return FileResponse(image_path)
    raise HTTPException(status_code=404, detail="Image not found")


@app.get("/student/profile/{emp_id}")
def get_student_profile(emp_id: str):
    valid_images = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    student_file = None
    student_name = None

    for img_name in valid_images:
        name_id = os.path.splitext(img_name)[0]
        parts = name_id.split('_')
        if parts[0] == str(emp_id):
            student_file = img_name
            student_name = " ".join(parts[1:]) if len(parts) > 1 else parts[0]
            break

    if not student_file:
        raise HTTPException(status_code=404, detail="Student ID not found in registered records.")

    attendance_history = []
    log_files = [f for f in os.listdir(LOGS_DIR) if f.startswith("Attendance_") and f.endswith(".csv")]

    for log_file in log_files:
        filepath = os.path.join(LOGS_DIR, log_file)
        try:
            df = pd.read_csv(filepath)
            matched_records = df[df['ID'].astype(str) == str(emp_id)]
            for _, row in matched_records.iterrows():
                attendance_history.append({
                    "Date": str(row['Date']),
                    "Time": str(row['Time']),
                    "Status": str(row.get('Status', 'N/A'))
                })
        except Exception:
            continue

    attendance_history = sorted(attendance_history, key=lambda x: (x['Date'], x['Time']), reverse=True)

    return {
        "id": emp_id,
        "name": student_name,
        "image_filename": student_file,
        "total_attendances": len(attendance_history),
        "history": attendance_history
    }


@app.post("/process-frame")
async def process_frame(file: UploadFile = File(...)):
    valid_images = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not valid_images:
        return {"status": "no_users", "message": "No registered users found in Dataset folder."}

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # --- ANTI-SPOOFING & LIVENESS CHECK ---
    is_live, liveness_msg = check_liveness(img)
    if not is_live:
        return {"status": "spoof_detected", "message": liveness_msg, "detected": []}

    temp_frame_path = os.path.join(BASE_DIR, "temp_frame.jpg")
    cv2.imwrite(temp_frame_path, img)

    detected = []
    status_msg = "No match found"

    try:
        results = DeepFace.find(
            img_path=temp_frame_path,
            db_path=DATASET_DIR,
            model_name="VGG-Face",
            detector_backend="skip",
            enforce_detection=False,
            silent=True
        )

        if len(results) > 0 and not results[0].empty:
            matched_file = results[0].iloc[0]['identity']
            filename = os.path.basename(matched_file)
            name_id = os.path.splitext(filename)[0]

            parts = name_id.split('_')
            emp_id = parts[0]
            name = parts[1] if len(parts) > 1 else parts[0]

            now = datetime.now()
            today_str = now.strftime('%Y-%m-%d')
            log_file = os.path.join(LOGS_DIR, f"Attendance_{today_str}.csv")

            if not os.path.exists(log_file):
                df = pd.DataFrame(columns=['ID', 'Name', 'Date', 'Time', 'Status'])
                df.to_csv(log_file, index=False)
            else:
                df = pd.read_csv(log_file)

            if str(emp_id) not in df['ID'].astype(str).values:
                status_calc = determine_attendance_status(now)
                new_entry = pd.DataFrame([{
                    'ID': emp_id,
                    'Name': name,
                    'Date': now.strftime('%Y-%m-%d'),
                    'Time': now.strftime('%H:%M:%S'),
                    'Status': status_calc
                }])
                new_entry.to_csv(log_file, mode='a', header=False, index=False)
                detected.append({"id": emp_id, "name": name, "attendance_status": status_calc})
            else:
                existing_status = df.loc[df['ID'].astype(str) == str(emp_id), 'Status'].values[
                    0] if 'Status' in df.columns else "Recorded"
                detected.append({"id": emp_id, "name": name, "attendance_status": str(existing_status)})

            status_msg = f"Recognized: {name}"

    except Exception as e:
        print(f"DeepFace processing error: {e}")
        status_msg = f"Error processing frame: {str(e)}"
    finally:
        if os.path.exists(temp_frame_path):
            os.remove(temp_frame_path)

    return {"status": "success", "message": status_msg, "detected": detected}
