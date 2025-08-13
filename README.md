#  Automated Deforestation Detection System

This project is a two-part system that **automatically collects satellite imagery** and uses **machine learning** to detect deforestation.

---

##  Overview

The system has **two main components**:

1. **Automated Data Collection**  
   A scheduled script runs daily (or at any chosen interval) to fetch the latest satellite image of a target location from **Google Earth Engine (GEE)**.  
   The image is automatically saved to your **Google Drive**, so you always have up-to-date data without manual downloads.

2. **Deforestation Detection**  
   A **Google Colab notebook** loads the newest image from Google Drive and runs a **pre-trained U-Net deep learning model** to detect deforested areas.  
   Results are displayed as an **image with a red overlay mask** highlighting cleared land.

---

##  Features

- **Fully Automated Image Collection** using the GEE API.
- **Scheduled Execution** via cron (Linux/Mac) or Task Scheduler (Windows).
- **Pre-trained U-Net Segmentation Model** for high accuracy.
- **Google Drive Integration** for seamless storage and retrieval.
- **Visual Alerts** showing deforestation zones in red.

---


---

##  Getting Started

### 1. Automated Data Collection

1. **Set up Google Earth Engine API**  
   - Create a GEE account: [https://earthengine.google.com](https://earthengine.google.com)  
   - Install the Earth Engine Python API:  
     ```bash
     pip install earthengine-api
     ```
   - Authenticate:
     ```bash
     earthengine authenticate
     ```

2. **Configure Script**  
   - Edit `config.json` with your target area coordinates and desired output path in Google Drive.

3. **Schedule the Script**  
   - **Linux/Mac (cron):**
     ```bash
     crontab -e
     # Example: run daily at 8 AM
     0 8 * * * python /path/to/fetch_satellite_images.py
     ```
   - **Windows (Task Scheduler)**: Add a daily task to run the script.

---

### 2. Deforestation Detection

1. **Open Google Colab** and upload the `deforestation_detection.ipynb` notebook.
2. **Mount Google Drive** inside Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
