
import os
import time
import threading
import traceback
from flask import Flask, render_template, request, jsonify, send_file, Blueprint, g
from reverseProxy import proxyRequest
from werkzeug.utils import secure_filename
from logging.config import dictConfig
from classifier import predict_nifti

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

AI_blueprint = Blueprint('AI', __name__)

@app.route('/')
@app.route('/<path:path>')
def index(path=''):
    if MODE == 'development':
        return proxyRequest(DEV_SERVER_URL, path)
    else:
        return render_template("index.html")
    
@app.route('/isAlive')
def isAlive():
    return jsonify({'error': '', 'message': 'Application is running'}), 200


@AI_blueprint.route('/predict', methods=['POST'])
def predict():
    app.logger.info('predict. upload file request received')

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

        save_path = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        g.savepath = save_path
        

        file.save(save_path)
        app.logger.info('File Received')

        result_dir = predict_nifti(save_path)
        g.resultdir = result_dir

        return send_file(result_dir, mimetype='application/nifti', as_attachment=True, download_name="result.nii.gz")

    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed due to {}'.format(e)}), 500
    
@AI_blueprint.after_request
def file_cleanup_thread(response):
    app.logger.info("file_cleanup_thread. Start thread to remove files")
    if response.status_code == 200:
        savepath = getattr(g, 'savepath', None)
        resultdir = getattr(g, 'resultdir', None)
        threading.Thread(target=file_cleanup, args=(savepath, resultdir,)).start()
    
    return response

def file_cleanup(savepath, resultdir):
    with app.app_context():
        app.logger.info("file_cleanup. Remove file after prediction")
        time.sleep(30)

        try:
            os.remove(savepath)
            os.remove(resultdir)
        except Exception as e:
            app.logger.error('Error deleting: {}'.format(e))

app.register_blueprint(AI_blueprint)