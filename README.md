
# Pneumonia Classification System

This project is designed to detect and classify pneumonia from medical images using deep learning models. It features a **Flask** backend that interacts with the **React** frontend, allowing users to upload chest X-ray images for analysis. The backend processes these images using a model and returns the classification results. The frontend allows users to interact with the system easily.

## Features

Pneumonia Detection: Classifies chest X-ray images to identify signs of pneumonia.

Transfer Learning: Utilizes pre-trained CNN models for efficient and accurate predictions.

Web Interface: User-friendly interface built with Flask for easy image uploads and result visualization.


## Project Structure
```bash
PNEUMOAI/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html         # HTML template for the web interface
├── static/
│   ├── css/               # CSS files
│   └── images/            # Sample images
├── models/                # Pre-trained CNN models
├── data/                  # Dataset (if included)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
 ```
## Prerequisites

Before you start, ensure the following are installed on your system:

1. **Node.js and npm**  
   Install [Node.js](https://nodejs.org/) (which includes npm) for the React frontend.

2. **Python**  
   Install Python 3.8 or higher from [python.org](https://www.python.org/).

3. **Flask**  
   Flask is required for the backend. You can install it via pip.

4. **Axios**  
   Axios is used for making HTTP requests between the frontend and backend.

---

## Project Setup

### 1. Backend Setup (Flask)

Follow these steps to set up the Flask backend:

1. Open a terminal and navigate to the `flask_backend` directory.

   ```bash
   cd flask_backend
   ```

2. Install the required dependencies using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

3. Add your **OpenAI API key** to the `app.py` file. Open `flask_backend/app.py` and find the section where the API key is needed. Replace it with your actual API key:

   ```python
   OPENAI_API_KEY = 'your-api-key-here'
   ```

4. Check if any path variables need to be changed in the `app.py` file to match your project directory structure.

5. After completing the above steps, start the Flask backend by running the following command:

   ```bash
   python app.py
   ```

   This will start the Flask server, and the backend will be live.

---

### 2. Frontend Setup (React)

Now, set up the frontend with React:

1. Open a new terminal and navigate to the `major_project` directory (which contains the React project).

   ```bash
   cd major_project
   ```

2. Install the necessary dependencies using npm:

   ```bash
   npm install
   ```

3. Start the React development server:

   ```bash
   npm run dev
   ```

   This will start the React frontend, and the application will be live at `http://localhost:3000`.

---

### 3. Using the Application

1. Navigate to the **Student Page** in the frontend.
2. Upload a chest X-ray image by clicking the upload button.
3. The backend will process the image, and after about a minute, the results will be displayed with the classification of the pneumonia type (if detected).


---

## Troubleshooting

1. **Backend not starting?**  
   - Ensure you have installed all dependencies with `pip install -r requirements.txt`.
   - Verify the OpenAI API key is correctly set in `app.py`.

2. **Frontend not loading?**  
   - Make sure you run `npm install` before starting the frontend.
   - Check the console for any missing dependencies or errors.

---

