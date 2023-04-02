import os
import io
import unittest
import tempfile
from flask import Request
from unittest.mock import MagicMock, patch
from app import app
from config import UPLOAD_DIR
from controllers.segmentation_controller import SegmentationController


class SegmentationControllerTest(unittest.TestCase):
    def setUp(self):    
        # create a test client
        self.client = app.test_client()
        self.app_context = app.app_context()
        self.app_context.push()
        
        self.controller = SegmentationController(app)

        self.test_file_path = os.path.abspath(os.path.join(UPLOAD_DIR, 'test.nii.gz'))
        
    def tearDown(self):
        # remove the app context after testing
        self.app_context.pop()
        
    def test_is_alive(self):
        response = self.client.get('/isAlive')
        assert response.status_code == 200

    def test_predict_file_not_found(self):
        request = MagicMock(spec=Request)
        request.files = {}

        response = self.client.post('/segmentation/predict')

        self.assertEqual(response.status_code, 400)
        self.assertIn(b'No file uploaded', response.get_data())

    def test_predict_file_not_allowed(self):
        response = self.client.post('/segmentation/predict', data=dict(
                               file=(io.BytesIO(b"this is a test"), 'test.pdf')))


        self.assertEqual(response.status_code, 400)
        self.assertIn(b'Only NIfTI files are supported', response.get_data())

    @patch('controllers.segmentation_controller.SegmentationService.predict_nifti')
    def test_predict_file_nifti(self, mock_predict_nifti):
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        mock_predict_nifti.return_value = temp_file.name

        response = self.client.post('/segmentation/predict', data=dict(
                               file=(io.BytesIO(b"this is a test"), 'test.nii.gz')))
        
        self.assertEqual(response.mimetype, 'application/nifti')
        self.assertEqual(response.headers['Content-Disposition'], 'attachment; filename=result.nii.gz')

    @patch('controllers.segmentation_controller.SegmentationService.predict_nifti')
    def test_predict_exception(self, mock_predict_nifti):
        mock_predict_nifti.side_effect = Exception('Test Exception')

        response = self.client.post('/segmentation/predict', data=dict(
                               file=(io.BytesIO(b"this is a test"), 'test.nii.gz')))

        self.assertEqual(response.status_code, 500)
        self.assertIn(b'Failed due to Test Exception', response.get_data())


if __name__ == '__main__':
    unittest.main()