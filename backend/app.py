
import os
from flask import Flask, render_template, request
from reverseProxy import proxyRequest
from classifier import classifyImage
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
# UPLOAD_FOLDER = '\\images'
# ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])    

app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ignore static folder in development mode.
if MODE == "development":
    app = Flask(__name__, static_folder=None)

@app.route('/')
@app.route('/<path:path>')
def index(path=''):
    if MODE == 'development':
        return proxyRequest(DEV_SERVER_URL, path)
    else:
        return render_template("index.html")    


@app.route('/classify', methods=['POST'])
def classify():
    app.logger.info('request value %s', request)
    if (request.files['image']): 
        file = request.files['image']

        result = classifyImage(file)
        print('Model classification: ' + result)     
        # filename = secure_filename(file.filename)
        # file.save(os.path.join(UPLOAD_FOLDER, filename))   
        return result