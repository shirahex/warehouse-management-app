
# Carton Compliance Analyzer 📦

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-orange.svg)](https://streamlit.io)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-blueviolet.svg)](https://ultralytics.com/)
<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) -->

A Streamlit web application designed for analyzing carton compliance in images. It uses a YOLOv8 object detection model to identify cartons and then applies a series of configurable analysis modules to assess stacking quality, overhang issues, zone-based counts, and overall compliance.

This application is part of the `warehouse-management-app` repository.

## ✨ Features

*   **Object Detection:** Utilizes the included YOLOv8 model (`weights/best.pt`) to detect cartons in uploaded images.
*   **Pairwise Overhang Analysis:** Identifies cartons improperly overhanging others based on configurable thresholds.
*   **Advanced Stack Analysis:**
    *   Identifies vertical stacks of cartons.
    *   Checks for maximum stack height compliance.
    *   Analyzes stack alignment and stability.
*   **Zone-Based Counting:** Allows users to define custom rectangular zones on the image and counts cartons within each.
*   **Compliance Scoring:** Generates an overall compliance score based on the various analysis checks.
*   **Interactive UI:**
    *   Image upload for analysis.
    *   Visualization of detected boxes, overhangs, stacks, and zones directly on the image.
    *   Detailed results presented in tabs (Detections, Overhang, Stacks, Zones, Summary).
*   **Dashboard:** Provides an overview of analysis history, compliance trends, detection counts, and processing times.
*   **Analysis History:** Stores and allows review of past analysis results.
*   **Configurable Settings:**
    *   Detection parameters (confidence, IoU).
    *   Expected carton count.
    *   Enable/disable and configure parameters for overhang, stack, and zone analyses.
    *   Zone definitions (name, coordinates, color).
*   **Settings Persistence:** User settings and analysis history are saved locally in JSON files within the `data/` directory.
*   **Custom Theming:** Includes custom CSS for a polished look and feel.

## 📸 Screenshots

![image](https://github.com/user-attachments/assets/90f9a80c-6987-4edf-881e-92d7a902b472)


## 🛠️ Tech Stack

*   **Backend/ML:**
    *   Python 3.8+
    *   Ultralytics YOLOv8 (for object detection)
    *   OpenCV (for image processing)
    *   NumPy (for numerical operations)
    *   Pillow (PIL) (for image manipulation)
*   **Frontend/Web Framework:**
    *   Streamlit
*   **Data & Visualization:**
    *   Pandas (for data manipulation)
    *   Plotly (for interactive charts)
*   **Streamlit Extras:**
    *   `streamlit-option-menu` (for sidebar navigation)
    *   `streamlit-extras` (for UI enhancements like colored headers, metric cards)

##  Prerequisites

*   Python 3.8 or higher
*   Git

## ⚙️ Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/shirahex/warehouse-management-app.git
    cd warehouse-management-app
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    The required Python packages are listed in `requirements.txt`. The YOLO model (`weights/best.pt`) is included in this repository.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: The `packages.txt` file is typically used for system-level dependencies when deploying to platforms like Streamlit Cloud. For local setup, `requirements.txt` is usually sufficient.)*

## 🚀 Running the Application

Once the setup is complete, run the Streamlit application:

```bash
streamlit run app.py


The application should open in your default web browser (usually at http://localhost:8501).

☁️ Deployment (e.g., to Streamlit Cloud)

This application is structured to be easily deployable on platforms like Streamlit Cloud.

Ensure your requirements.txt file is up-to-date with all Python dependencies.

The packages.txt file can be used to specify system-level dependencies that need to be installed via apt-get (e.g., ffmpeg, libsm6, libxext6 if OpenCV needs them). Streamlit Cloud will automatically try to install these.

Connect your GitHub repository to Streamlit Cloud and deploy the app.py file.

📁 Project Structure
warehouse-management-app/
├── data/                       # Stores user data
│   ├── analysis_history.json   # History of analyses
│   └── user_settings.json      # User-defined settings
├── weights/
│   └── best.pt                 # Pre-trained YOLO model file
├── .gitignore                  # Specifies intentionally untracked files
├── app.py                      # Main Streamlit application script
├── packages.txt                # System-level dependencies for deployment
├── requirements.txt            # Python package dependencies
└── README.md                   # This file
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
🔧 Configuration

Most configurations can be done directly through the "Settings" page in the application UI. These settings are saved to data/user_settings.json.

Detection Parameters: Confidence threshold, IoU threshold for Non-Maximum Suppression (NMS).

Analysis Modules: Enable/disable and set parameters for:

Pairwise Overhang Analysis

Advanced Stack Analysis

Zone Counting

Zone Management: Define up to MAX_ZONES (currently 5) with names, pixel coordinates (Xmin, Ymin, Xmax, Ymax), and display colors.

Appearance: A conceptual dark mode toggle is present.

Data Management: Save, load, or reset settings to default.

🎨 Customization

Theme Colors: The THEME dictionary in app.py defines the color palette used throughout the app and in the custom CSS.

CSS: Custom CSS is loaded via the load_custom_css() function. You can modify the CSS rules within this function to further tailor the appearance.

YOLO Model: To use a different model, replace weights/best.pt or update the MODEL_PATH variable in app.py. If you replace the model, ensure it's compatible with the Ultralytics YOLO library version specified in requirements.txt.

📈 Future Enhancements

Live Camera Feed Analysis: Integrate real-time analysis from a camera stream.

Batch Image Processing: Allow users to upload and analyze multiple images at once.

Advanced Reporting: Generate PDF or CSV reports of the analysis.

User Authentication: For multi-user environments or to protect sensitive data.

Database Integration: Use a proper database (e.g., SQLite, PostgreSQL) instead of JSON files for more robust data storage, especially for analysis_history.

More Sophisticated Error Handling & Logging.

Unit and Integration Tests.

🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or bug fixes, please:

Fork the repository.

Create a new branch (git checkout -b feature/AmazingFeature).

Make your changes.

Commit your changes (git commit -m 'Add some AmazingFeature').

Push to the branch (git push origin feature/AmazingFeature).

Open a Pull Request.

📄 License

This project is currently unlicensed. Consider adding a license file (e.g., LICENSE.md with the MIT License text) to define how others can use your code.

🙏 Acknowledgements

Streamlit for the awesome web app framework.

Ultralytics for the YOLO object detection models and library.

Plotly for interactive charting.

The creators of streamlit-option-menu and streamlit-extras.

Happy Analyzing! 📦🚀
