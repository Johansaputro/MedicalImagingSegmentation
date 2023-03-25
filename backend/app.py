
import os
import traceback
from flask import Flask, render_template, request, jsonify, send_file, g
from reverseProxy import proxyRequest
from werkzeug.utils import secure_filename
from logging.config import dictConfig
from classifier import predict

dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "default",
            }
        },
        "root": {"level": "INFO", "handlers": ["console"]},
    }
)

MODE = os.getenv('FLASK_ENV')
DEV_SERVER_URL = 'http://localhost:3000/'   

app = Flask(__name__)
UPLOAD_FOLDER = os.path.abspath(os.path.dirname(__file__)) + '/receivedFile'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ignore static folder in development mode.
if MODE == "development":
    app = Flask(__name__, static_folder=None)
    UPLOAD_FOLDER = os.path.abspath(os.path.dirname(__file__)) + '/receivedFile'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
@app.route('/<path:path>')
def index(path=''):
    if MODE == 'development':
        return proxyRequest(DEV_SERVER_URL, path)
    else:
        return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    app.logger.info('upload file request received')

    if 'file' not in request.files:
        app.logger.error('No File Uploaded')
        return jsonify({'error': 'No file uploaded.'}), 400

    file = request.files['file']

    # Save the file to disk
    try:
        filename = secure_filename(file.filename)
        if not filename.endswith('.nii.gz'):
            app.logger.error('Not a NIfTI file')
            return jsonify({'error': 'Only NIfTI files are supported.'}), 400

        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        g.savepath = save_path
        

        file.save(save_path)
        app.logger.info('File Received')

        result_dir = predict(save_path)
        g.resultdir = result_dir

        return send_file(result_dir, mimetype='application/octet-stream', as_attachment=True, attachment_filename="result.nii.gz")

    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed due to {}'.format(e)}), 500
    
@app.after_request
def file_cleanup():

    save_path = getattr(g, 'savepath', None)
    result_dir = getattr(g, 'resultdir', None)

    try:
        os.remove(save_path)
        os.remove(result_dir)
    except Exception as e:
        app.logger.error('Error deleting: {}'.format(e))