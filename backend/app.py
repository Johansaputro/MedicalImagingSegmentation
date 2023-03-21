
import os
import traceback
from flask import Flask, render_template, request
from reverseProxy import proxyRequest
from werkzeug.utils import secure_filename
from logging.config import dictConfig

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

@app.route('/upload', methods=['POST'])
def upload():
    app.logger.info('upload file request received with request : %s', request)
    if 'file' not in request.files:
        return 'No file uploaded', 400

    file = request.files['file']

    # Save the file to disk
    try:
        filename = file.filename
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        file.save(save_path)
        app.logger.info('File Received')
        return "File Uploaded Succesfully"

    except Exception as e:
        app.logger.error(traceback.format_exc())
        return "Failed due to {}".format(e)
        

    # Call some other function that needs the file
    # some_function(save_path)

    # Delete the file after a delay
    # os.remove(save_path)
    # app.logger.info('File Removed')   


# @app.route('/classify', methods=['POST'])
# def classify():
#     app.logger.info('request value %s', request)
#     if (request.files['image']): 
#         file = request.files['image']

#         result = classifyImage(file)
#         print('Model classification: ' + result)     
#         # filename = secure_filename(file.filename)
#         # file.save(os.path.join(UPLOAD_FOLDER, filename))   
#         return result