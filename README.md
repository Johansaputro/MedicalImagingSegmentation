## Overview
Final University Project - Medical Imaging Segmentation  
Start both FE and BE  :
1. Go to frontend folder
2. npm run start:server-dev  
  
use this commit message for every commit -> "(Name) - (Commit Description)"  
e.g $git commit -m "Budi - Initial Commit"  


## Backend Folder (On Development)

Uses Python 3.8 and Flask as Microservices  
Database Used will be either SQL/MongoDB  

Required Libraries:  
1. pip install requirements.txt

How to start backend only:  
1. go to backend folder  
2. Enter $python app.py    on the terminal  

How to start backend only using Docker: 
1. docker build -t medical-segmentation-app .  
2. docker run  -d -p 5000:5000 medical-segmentation-app  

-----------------  

## Frontend Folder (On Development)  

Uses ReactJs  
Please read the quick start guide at frontend/README.MD  
To Migrate to NextJs please follow these following link https://www.geeksforgeeks.org/how-to-migrate-from-create-react-app-to-next-js/  



## Sources and Reference

Please add sources and references here  
https://medium.com/sopra-steria-norge/build-a-simple-image-classification-app-using-react-keras-and-flask-7b9075e3b6f5  

AI Model, preprocessing, postprocessing references (temporary):  
https://github.com/assassint2017/abdominal-multi-organ-segmentation  

Dataset  (need to create account first to see abdomen dataset):  
https://www.synapse.org/#!Synapse:syn3193805/wiki/217752  