import os
import time
import uuid
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import torch
import cv2
from PIL import Image

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload size

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_unique_filename(filename):
    """Generate a unique filename to prevent overwrites"""
    unique_id = uuid.uuid4().hex[:8]
    name, ext = os.path.splitext(filename)
    return f"{name}_{unique_id}{ext}"

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Create unique filename
            filename = secure_filename(file.filename)
            unique_filename = generate_unique_filename(filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save the file
            file.save(filepath)
            
            # Start processing timer
            start_time = time.time()
            
            # Video processing
            if unique_filename.lower().endswith(('.mp4', '.avi', '.mov')):
                # Create unique result filenames
                results_video_name = f"result_{unique_filename}"
                plain_video_name = f"plain_{unique_filename}"
                
                results_video_path = os.path.join(app.config['RESULTS_FOLDER'], results_video_name)
                plain_video_path = os.path.join(app.config['RESULTS_FOLDER'], plain_video_name)
                
                # Process video
                cap = cv2.VideoCapture(filepath)
                
                # Get video properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Create video writers
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_results = cv2.VideoWriter(results_video_path, fourcc, fps, (width, height))
                out_plain = cv2.VideoWriter(plain_video_path, fourcc, fps, (width, height))
                
                detected_objects_set = set()
                frame_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every 3rd frame for faster processing (adjust as needed)
                    if frame_count % 3 == 0:
                        # Convert BGR to RGB
                        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        
                        # Run detection
                        results = model(img)
                        
                        # Get annotated frame
                        annotated_frame = results.render()[0]
                        out_results.write(annotated_frame)
                        
                        # Collect detected objects
                        predictions = results.pred[0].tolist()
                        for pred in predictions:
                            detected_objects_set.add(results.names[int(pred[5])])
                    else:
                        out_results.write(frame)
                    
                    # Write original frame to plain video
                    out_plain.write(frame)
                    frame_count += 1
                
                cap.release()
                out_results.release()
                out_plain.release()
                
                # Convert set to sorted list
                detected_objects = sorted(list(detected_objects_set))
                
                # Calculate processing time
                processing_time = round(time.time() - start_time, 2)
                
                return redirect(url_for('uploaded_file', 
                                       filename=results_video_name,
                                       plain_filename=plain_video_name,
                                       objects=detected_objects,
                                       processing_time=processing_time))
            
            # Image processing
            else:
                # Create unique result filename
                results_image_name = f"result_{unique_filename}"
                results_image_path = os.path.join(app.config['RESULTS_FOLDER'], results_image_name)
                
                # Open and process image
                img = Image.open(filepath)
                results = model(img)
                
                # Save annotated image
                results_img = results.render()[0]
                cv2.imwrite(results_image_path, results_img)
                
                # Collect detected objects
                predictions = results.pred[0].tolist()
                detected_objects_set = set()
                for pred in predictions:
                    detected_objects_set.add(results.names[int(pred[5])])
                
                # Convert set to sorted list
                detected_objects = sorted(list(detected_objects_set))
                
                # Calculate processing time
                processing_time = round(time.time() - start_time, 2)
                
                return redirect(url_for('uploaded_file', 
                                       filename=results_image_name,
                                       objects=detected_objects,
                                       processing_time=processing_time))
    
    return render_template('upload.html')

@app.route('/uploads')
def uploaded_file():
    filename = request.args.get('filename')
    plain_filename = request.args.get('plain_filename')
    detected_objects = request.args.getlist('objects')
    processing_time = request.args.get('processing_time', '0')
    
    return render_template('uploaded.html', 
                          filename=filename, 
                          plain_filename=plain_filename, 
                          objects=detected_objects,
                          processing_time=processing_time)

# Add a route to clean up old results
@app.route('/cleanup')
def cleanup():
    # This could be called periodically to clean up old results
    for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) and (time.time() - os.path.getmtime(file_path)) > 3600:  # 1 hour old
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    return redirect(url_for('upload_file'))

if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    app.run(debug=True)