# YOLOv5 Object Detection Web Application

This is a Flask-based web application that allows users to upload images or videos for object detection using the YOLOv5 model. The application processes the uploaded files and displays the results along with the detected objects.

## Features

- Upload images (PNG, JPG, JPEG) or videos (MP4, AVI, MOV).
- Process uploaded files using the YOLOv5 object detection model.
- Display results with annotated images or videos showing detected objects.
- Provide a list of detected objects.

## Requirements

- Python 3.7 or higher
- Flask
- PyTorch
- OpenCV
- Pillow

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the following directories exist:
   ```bash
   mkdir uploads static/results
   ```

## Running the Application

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open a web browser and go to `http://127.0.0.1:5000/`.

## Usage

1. On the main page, upload an image or video file.
2. The application will process the file and redirect to the results page.
3. The results page will display the processed file with annotated detections and list the detected objects.

## File Structure

- `app.py`: The main Flask application file.
- `templates/upload.html`: The HTML template for the file upload page.
- `templates/uploaded.html`: The HTML template for the results page.
- `uploads/`: Directory to store uploaded files.
- `static/results/`: Directory to store processed files and results.

## Code Explanation

- **Import Statements**: Import necessary libraries and modules.
- **Configuration**: Define upload and results folders and allowed file extensions.
- **YOLOv5 Model Loading**: Load the YOLOv5 model using PyTorch Hub.
- **Utility Functions**:
  - `allowed_file(filename)`: Check if the file has an allowed extension.
- **Routes**:
  - `upload_file()`: Handle file upload and processing.
  - `uploaded_file()`: Display the processed file and detected objects.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Feel free to customize this README to suit your specific application and repository details.
