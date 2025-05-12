How to Run the Customer Campaign Prediction App on MacOS
=======================================================

1. Install Python (if not already installed)
-------------------------------------------
- Download and install Python 3.8 or newer from https://www.python.org/downloads/
- After installation, verify by running:
  python3 --version

2. Open Terminal and Navigate to the Project Directory
------------------------------------------------------
- Use the cd command to go to the folder containing app.py and marketing_campaign.xlsx.
  Example:
    cd /path/to/your/project

3. (Recommended) Create a Virtual Environment
---------------------------------------------
- python3 -m venv venv
- source venv/bin/activate

4. Install Required Python Packages
-----------------------------------
- pip install streamlit pandas scikit-learn openpyxl

5. Run the Streamlit App
------------------------
- streamlit run app.py
- The terminal will show a local URL (e.g., http://localhost:8501). Open this URL in your browser.

6. Using the App
----------------
- Enter values for the displayed top features and click 'Predict Response'.
- The app will show whether the customer will respond to the campaign and provide reasons based on your input.

Notes:
------
- Make sure marketing_campaign.xlsx is in the same folder as app.py.
- If you get a permissions error, try: chmod +x app.py
- If you encounter issues with package installation, try using pip3 instead of pip.

If you need further help, contact your developer or AI assistant!
