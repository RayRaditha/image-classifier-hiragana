from process import *

from flask import Flask, render_template, request, send_from_directory
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
app.config['PROCESSED_FOLDER'] = './static/processed/'

# Ensure the directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        if request.method == 'POST':
            if request.files:
                image = request.files['image']
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
                image.save(img_path)

                # Get prediction and accuracy
                prediction, accuracy = predict_process(img_path)

                # Save processed image
                processed_img_filename = f"processed_{image.filename}"
                processed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_img_filename)
                processed_image = import_image(img_path)
                prep_image(processed_image, save_path=processed_image_path)

                # return render_template('index.html', 
                #                         uploaded_image=image.filename,  
                #                         prediction=prediction, 
                #                         accuracy=accuracy)
                return render_template('index.html', 
                                       uploaded_image=image.filename, 
                                       processed_image=processed_img_filename, 
                                       prediction=prediction, 
                                       accuracy=accuracy)
    except Exception as e:
        print("An error occurred:", str(e))
        # Print the full traceback
        import traceback
        traceback.print_exc()

    return render_template('index.html')

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/display_processed/<filename>')
def send_processed_image(filename=''):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)