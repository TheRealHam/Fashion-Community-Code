from flask import Flask, render_template, request, redirect  # noqa: F401
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Pretrained model setup
model = resnet18(pretrained=True)
model.eval()

# Load ImageNet class labels
with open('imagenet_classes.txt') as f:
    class_labels = [line.strip() for line in f.readlines()]

# Transformation for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Utility function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define routes
@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Classifier</title>
    </head>
    <body>
        <h1>Upload an Image</h1>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <button type="submit">Upload</button>
        </form>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Perform classification
        label = classify_image(filepath)
        os.remove(filepath)  # Clean up uploaded file
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Classification Result</title>
        </head>
        <body>
            <h1>Classification Result</h1>
            <p>Label: {label}</p>
            <a href="/">Classify another image</a>
        </body>
        </html>
        '''

    return "File not allowed"

def classify_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = outputs.max(1)
        label = class_labels[predicted.item()]
    return label

if __name__ == '__main__':
    app.run(debug=True)