import warnings
warnings.filterwarnings('ignore')

import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from datetime import datetime
from deepface import DeepFace


# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Recognition Attendance",
    page_icon="👤",
    layout="wide"
)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(__file__)
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")
DB_PATH         = KNOWN_FACES_DIR


# ── Helper Functions ──────────────────────────────────────────────────────────
def get_registered_people():
    """Returns list of registered people from known_faces folder."""
    people = []
    if os.path.exists(KNOWN_FACES_DIR):
        for f in os.listdir(KNOWN_FACES_DIR):
            if f.endswith(('.jpg', '.jpeg', '.png')):
                people.append(os.path.splitext(f)[0])
    return people


def mark_attendance(name):
    """
    Marks attendance for a recognized person.
    Avoids duplicate entries for the same person on the same day.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    now   = datetime.now().strftime("%H:%M:%S")

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Time", "Status"])

    already_marked = ((df["Name"] == name) & (df["Date"] == today)).any()

    if not already_marked:
        new_row = pd.DataFrame([{
            "Name":   name,
            "Date":   today,
            "Time":   now,
            "Status": "Present"
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        os.makedirs(os.path.dirname(ATTENDANCE_FILE), exist_ok=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        return True
    return False


def crop_black_borders(image):
    """Removes black borders from an image automatically."""
    bbox = image.getbbox()
    if bbox:
        return image.crop(bbox)
    return image


def recognize_face(image_path):
    """
    Uses DeepFace to find a match in the known_faces folder.
    Returns the matched name or 'Unknown'.
    """
    try:
        results = DeepFace.find(
            img_path          = image_path,
            db_path           = DB_PATH,
            enforce_detection = False,
            silent            = True
        )
        if results and len(results[0]) > 0:
            match_path = results[0].iloc[0]['identity']
            name = os.path.splitext(os.path.basename(match_path))[0]
            return name
    except Exception:
        pass
    return "Unknown"


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/face-id.png", width=80)
st.sidebar.title("Attendance AI")
st.sidebar.markdown("Face Recognition Attendance System")
st.sidebar.divider()

people = get_registered_people()
st.sidebar.markdown(f"**Registered:** {len(people)} people")
for person in people:
    st.sidebar.markdown(f"- {person}")

page = st.sidebar.radio("Navigate", ["Take Attendance", "Register Face", "View Records"])


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — TAKE ATTENDANCE
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Take Attendance":
    st.title("Take Attendance")
    st.markdown("Upload a photo or use your camera to mark attendance automatically")

    if not people:
        st.warning("No registered faces found. Please register faces first.")
    else:
        method     = st.radio("Input Method", ["Upload Image", "Use Camera"])
        image_file = None

        if method == "Upload Image":
            image_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
        else:
            image_file = st.camera_input("Take a photo")

        if image_file:
            temp_path = os.path.join(BASE_DIR, "temp_input.jpg")
            image     = Image.open(image_file).convert("RGB")
            image     = crop_black_borders(image)
            image.save(temp_path)

            st.image(image, caption="Input Photo", use_container_width=True)

            with st.spinner("Recognizing face..."):
                name = recognize_face(temp_path)

            st.divider()

            if name != "Unknown":
                marked = mark_attendance(name)
                if marked:
                    st.success(f"Attendance marked for **{name}**")
                else:
                    st.info(f"**{name}** already marked today")
            else:
                st.error("Face not recognized. Please register first.")

            if os.path.exists(temp_path):
                os.remove(temp_path)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — REGISTER FACE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Register Face":
    st.title("Register New Face")
    st.markdown("Add a new person to the system")

    name       = st.text_input("Full Name")
    method     = st.radio("Photo Method", ["Upload Photo", "Use Camera"])
    image_file = None

    if method == "Upload Photo":
        image_file = st.file_uploader("Upload a clear face photo", type=["jpg", "jpeg", "png"])
    else:
        image_file = st.camera_input("Take a photo")

    if image_file and name:
        image = Image.open(image_file).convert("RGB")
        image = crop_black_borders(image)
        st.image(image, caption=f"Preview: {name}", width=300)

        if st.button("Register", use_container_width=True):
            os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
            save_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
            image.save(save_path)
            st.success(f"{name} registered successfully! Please refresh the page.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — VIEW RECORDS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "View Records":
    st.title("Attendance Records")
    st.markdown("View and filter all attendance records")

    if not os.path.exists(ATTENDANCE_FILE):
        st.info("No attendance records yet.")
    else:
        df = pd.read_csv(ATTENDANCE_FILE)

        col1, col2 = st.columns(2)
        with col1:
            selected_date = st.selectbox("Filter by Date",
                ["All"] + sorted(df["Date"].unique().tolist(), reverse=True))
        with col2:
            selected_name = st.selectbox("Filter by Name",
                ["All"] + sorted(df["Name"].unique().tolist()))

        filtered_df = df.copy()
        if selected_date != "All":
            filtered_df = filtered_df[filtered_df["Date"] == selected_date]
        if selected_name != "All":
            filtered_df = filtered_df[filtered_df["Name"] == selected_name]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(filtered_df))
        col2.metric("Unique People", filtered_df["Name"].nunique())
        col3.metric("Today",
            len(filtered_df[filtered_df["Date"] == datetime.now().strftime("%Y-%m-%d")]))

        st.divider()
        st.dataframe(filtered_df, use_container_width=True)

        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label     = "Download CSV",
            data      = csv,
            file_name = "attendance_records.csv",
            mime      = "text/csv"
        )