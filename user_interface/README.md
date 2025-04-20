# Streamlit Waste Classification App
This Streamlit web app enables users to upload images of waste items for real-time classification into appropriate waste categories. It also provides helpful recycling or disposal guidance based on the prediction.

## Instructions to run GUI code 

### 1. Downloading folder
Download this **user_interface** folder from this repository to your local storage, and navigate to its directory (e.g. `cd Downloads/user_interface`).


### 2. Creating a virtual environment
Within the folder, create a virtual environment with [**Python 3.11.1**](https://www.python.org/downloads/release/python-3111/)
```
python3.11 -m venv venv
```

...and activate it (re-activate for every session).

- MacOS:
    ```
    source venv/bin/activate
    ```

- Windows:
    ```
    venv/Scripts/activate
    ```


### 3. Installing dependencies
Install all required libraries and dependencies within the environment.
```
pip install -r requirements.txt
```

### 4. Downloading our model
Download our best performing classification model from (TO INSERT LINK), and insert it into the folder.

### 5. Running the webpage
Run the following command to start the Streamlit app.
```
streamlit run app.py
```
If the execution is successful, you should be automatically directed to the running application on your default web browser 👍

If it doesn't open automatically, visit `http://localhost:8501` in your browser.
