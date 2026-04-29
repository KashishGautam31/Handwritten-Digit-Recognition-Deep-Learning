import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from streamlit_drawable_canvas import st_canvas

# -----------------------------
# LOAD MODEL
# -----------------------------
model = load_model("model/digit_model.h5")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Digit Recognizer", layout="wide")

# -----------------------------
# CUSTOM CSS (BLUE + TRANSPARENT)
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #1e3c72, #2a5298);
    color: white;
}

/* Sidebar Blue */
section[data-testid="stSidebar"] {
    background: linear-gradient(to bottom, #1e3c72, #2a5298);
}

/* Remove boxes */
.block-container {
    background: transparent;
}

/* Buttons */
.stButton>button {
    background-color: #4B8BFF;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("✍️ Handwritten Digit Recognition")

# -----------------------------
# HISTORY
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# SIDEBAR (INSTRUCTIONS BACK)
# -----------------------------
st.sidebar.title("📌 Instructions")

st.sidebar.markdown("""
### How to Use:

1️⃣ Go to **Upload Tab** OR **Draw Tab**  
2️⃣ Upload image OR draw digit  
3️⃣ Click **Predict**  
4️⃣ View result + graph  

### Tips:
✔ Draw thick digit  
✔ Keep center aligned  
✔ Use white color  

""")

# -----------------------------
# GRAPH FUNCTION (VERTICAL FIXED)
# -----------------------------
def show_graph(pred):

    digits = list(range(10))
    confidence = pred[0] * 100

    fig = go.Figure(data=[
        go.Bar(
            x=digits,
            y=confidence,
            width=0.6,  # FIX: thickness
            marker=dict(
                color=[
                    '#FF4B4B','#4B8BFF','#00C49F','#FFBB28','#FF8042',
                    '#8E44AD','#2ECC71','#F39C12','#E74C3C','#1ABC9C'
                ]
            ),
            text=[f"{c:.1f}%" for c in confidence],
            textposition='outside'
        )
    ])

    fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Digits",
        yaxis_title="Confidence (%)",
        yaxis=dict(range=[0,100]),
        bargap=0.25,  # FIX: spacing
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📤 Upload", "✍️ Draw", "📜 History"])

# -----------------------------
# TAB 1 - UPLOAD
# -----------------------------
with tab1:

    uploaded_file = st.file_uploader("Upload Digit", type=["png","jpg","jpeg"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert('L')
        st.image(image, width=200)

        if st.button("🔍 Predict Upload"):

            img = image.resize((28,28))
            img = np.array(img)

            img = 255 - img
            img = img / 255.0
            img = img.reshape(1,28,28,1)

            pred = model.predict(img)
            digit = np.argmax(pred)

            st.success(f"Prediction: {digit}")

            st.session_state.history.append(f"Upload → {digit}")

            show_graph(pred)

# -----------------------------
# TAB 2 - DRAW (FIXED CANVAS)
# -----------------------------
with tab2:

    st.subheader("Draw your digit below 👇")

    # DEBUG TEXT (helps confirm rendering)
    st.write("Canvas loading...")

    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas_FIX_123",  # IMPORTANT FIX (force rerender)
    )

    st.write("Canvas loaded")

    if st.button("🧠 Predict Drawing"):

        if canvas_result.image_data is not None:

            img = canvas_result.image_data[:, :, 0]
            img = cv2.resize(img, (28,28))

            img = img / 255.0
            img = img.reshape(1,28,28,1)

            pred = model.predict(img)
            digit = np.argmax(pred)

            st.success(f"Prediction: {digit}")

            st.session_state.history.append(f"Draw → {digit}")

            show_graph(pred)

        else:
            st.warning("Draw something first!")

# -----------------------------
# TAB 3 - HISTORY
# -----------------------------
with tab3:

    st.subheader("📜 Prediction History")

    if len(st.session_state.history) == 0:
        st.info("No predictions yet")

    else:
        for i, item in enumerate(st.session_state.history, 1):
            st.write(f"{i}. {item}")