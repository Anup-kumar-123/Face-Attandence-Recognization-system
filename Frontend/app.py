import streamlit as st
import requests
import pandas as pd
import cv2
import time

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Smart Attendance System",
    page_icon="👤",
    layout="wide"
)

# --- MODERN CLEAN STYLING ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    div[data-testid="stMetricValue"] {
        color: #38BDF8;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #0EA5E9;
        color: #FFFFFF;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #0284C7;
    }
    .card {
        background-color: #1E293B;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("👤 Smart Face Recognition System")

tab1, tab2, tab3, tab4 = st.tabs([
    "📹 Live Attendance",
    "📋 Attendance Logs",
    "➕ New Registration",
    "👥 Student Profiles & Search"
])

# ----------------- TAB 1: LIVE WEBCAM SCANNER -----------------
with tab1:
    st.subheader("Real-Time Face Recognition & Anti-Spoofing Scanner")
    run_cam = st.checkbox("Turn On Camera Scanner")

    col1, col2 = st.columns([2, 1])
    with col1:
        FRAME_WINDOW = st.image([])
    with col2:
        st.write("### Live Detection Status")
        status_box = st.empty()
        status_box.info("Camera turned off. Check the box above to start.")

    if run_cam:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame_counter = 0

        while run_cam:
            success, frame = cap.read()
            if not success:
                status_box.error("Error: Unable to connect to webcam.")
                break

            frame = cv2.flip(frame, 1)

            frame_counter += 1
            if frame_counter % 6 == 0:
                _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/process-frame",
                        files={"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")},
                        timeout=2.0
                    )
                    if response.status_code == 200:
                        data = response.json()
                        status_type = data.get("status", "")
                        detected = data.get("detected", [])
                        msg = data.get("message", "")

                        if status_type == "spoof_detected":
                            status_box.error(f"🚨 {msg}")
                        elif detected:
                            user_status = detected[0].get('attendance_status', 'Recorded')
                            status_box.success(
                                f"Recognized: **{detected[0]['name']}** (ID: {detected[0]['id']})\n\n"
                                f"📌 **Status:** `{user_status}`"
                            )
                        elif "No match found" in msg:
                            status_box.warning("Unknown face detected. Not in database.")
                        else:
                            status_box.info("Scanning for registered face...")
                except requests.exceptions.Timeout:
                    pass
                except Exception:
                    status_box.error("Backend server offline.")

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            time.sleep(0.01)

        cap.release()
    else:
        FRAME_WINDOW.empty()

# ----------------- TAB 2: DAILY ATTENDANCE LOGS -----------------
with tab2:
    st.subheader("Attendance Log Manager")

    available_dates = []
    try:
        res = requests.get(f"{BACKEND_URL}/attendance/dates")
        if res.status_code == 200:
            available_dates = res.json().get("dates", [])
    except Exception as e:
        st.error(f"Error fetching available log dates: {e}")

    if available_dates:
        c1, c2 = st.columns([2, 1])
        with c1:
            selected_date = st.selectbox("Select Attendance Date:", available_dates)

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            fetch_btn = st.button("Fetch Records", use_container_width=True)
        with col_btn2:
            delete_btn = st.button("Delete Date Log", use_container_width=True)

        if fetch_btn and selected_date:
            try:
                res = requests.get(f"{BACKEND_URL}/attendance/by-date", params={"date": selected_date})
                if res.status_code == 200:
                    data = res.json()
                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info(f"No records found for {selected_date}.")
            except Exception as e:
                st.error(f"Error fetching data: {e}")

        if delete_btn and selected_date:
            try:
                res = requests.delete(f"{BACKEND_URL}/attendance/delete", params={"date": selected_date})
                if res.status_code == 200:
                    st.success(res.json().get("message"))
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Failed to delete log file.")
            except Exception as e:
                st.error(f"Deletion error: {e}")
    else:
        st.info("No saved attendance logs found.")

# ----------------- TAB 3: USER REGISTRATION -----------------
with tab3:
    st.subheader("Register New Profile")
    with st.form("registration_form"):
        emp_id = st.text_input("Student / Employee ID")
        name = st.text_input("Full Name")
        uploaded_file = st.file_uploader("Upload Clear Front-Facing Profile Photo", type=["jpg", "jpeg", "png"])
        submit = st.form_submit_button("Submit Registration")

        if submit:
            if emp_id and name and uploaded_file:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {"emp_id": emp_id, "name": name}

                try:
                    res = requests.post(f"{BACKEND_URL}/register", data=data, files=files)
                    if res.status_code == 200:
                        st.success(res.json().get("message"))
                    else:
                        st.error("Registration failed on server.")
                except Exception as e:
                    st.error(f"Error connecting to backend: {e}")
            else:
                st.warning("Please fill in all details and upload an image.")

# ----------------- TAB 4: STUDENT SEARCH & DIRECTORY -----------------
with tab4:
    st.subheader("Search Student Profile")

    c_search, c_btn = st.columns([3, 1])
    with c_search:
        search_id = st.text_input("Enter Student ID to search:", placeholder="e.g. 101", label_visibility="collapsed")
    with c_btn:
        search_btn = st.button("Search Profile", use_container_width=True)

    if search_btn and search_id:
        try:
            res = requests.get(f"{BACKEND_URL}/student/profile/{search_id.strip()}")
            if res.status_code == 200:
                profile = res.json()
                st.markdown("---")

                col_img, col_info = st.columns([1, 2])
                with col_img:
                    img_url = f"{BACKEND_URL}/students/image/{profile['image_filename']}"
                    st.image(img_url, caption=f"ID: {profile['id']}", width=220)

                with col_info:
                    st.markdown(f"## {profile['name']}")
                    st.markdown(f"**Student ID:** `{profile['id']}`")
                    st.metric("Total Attendance Days Logged", profile['total_attendances'])

                    st.markdown("### Attendance Logs")
                    if profile['history']:
                        history_df = pd.DataFrame(profile['history'])
                        st.dataframe(history_df, use_container_width=True)
                    else:
                        st.info("No attendance activity recorded yet.")
                st.markdown("---")
            else:
                st.error(f"Student ID '{search_id}' not found.")
        except Exception as e:
            st.error(f"Failed to load profile details: {e}")

    st.subheader("All Registered Students")
    try:
        res = requests.get(f"{BACKEND_URL}/students")
        if res.status_code == 200:
            students = res.json().get("students", [])
            if students:
                cols = st.columns(4)
                for index, student in enumerate(students):
                    with cols[index % 4]:
                        img_url = f"{BACKEND_URL}/students/image/{student['image_filename']}"
                        st.image(img_url, use_container_width=True)
                        st.markdown(f"**Name:** {student['name']}")
                        st.markdown(f"**ID:** `{student['id']}`")
                        st.markdown("---")
            else:
                st.info("No registered students found in dataset.")
    except Exception as e:
        st.error(f"Error connecting to server directory: {e}")
