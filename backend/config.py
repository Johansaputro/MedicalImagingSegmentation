import os
import logging
import torch
from Net import model

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

pred_dir = "/PredictResult"
RESULT_DIR = os.path.abspath(ROOT_DIR + pred_dir)
UPLOAD_DIR = os.path.abspath(os.path.dirname(__file__) + '/receivedFile')

logger.info("Loading Model")
weight_dir = "/Net/50-0.539-0.644.pth"
state_dict_dir = os.path.abspath(ROOT_DIR + weight_dir)
NETWORK =  model.Net(training=False)
NETWORK.load_state_dict(torch.load(state_dict_dir, map_location=torch.device('cpu')))
NETWORK.eval()
logger.info("Model Finished Loading")