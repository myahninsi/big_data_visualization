STEPS PERFORMED FOR THE PROJECT

1. CREATE A VIRTUAL ENVIRONMENT
py -3 -m venv .venv

2. INITATE THE VIRTUAL ENVIRONMENT
.venv\Scripts\activate 

3. DEFINE PYTHON VERSION
python 3.11.2

4. INSTALLING FLASK IN VIRTUAL ENVIRONMENT
pip install Flask==3.0.3

5. INSTALLING OTHER PYTHON LIBRARIES
pip install pandas==2.2.2
pip install matplotlib==3.9.1.post1
pip install seaborn==0.13.2
pip install numpy==2.0.1
pip install scikit-learn==1.4.2
pip install scipy==1.14.0
pip install openyxl==3.1.5

6. ESTABLISH THE FLASK STRUCTURE
* app.py : python file that manages the library flask and request from the web application
* predictor.py : python file that manages the model and returns predictions
* templates : folder that contains HTML files 
* static : folder that contains CSS files and images necessary for the project 
* pickle : folder that contains a pickle file where it can be retrieved the model
* data : folder that can contains excel or csv files necessary for the project

7. POWERBI DASHBOARD IN HTML
* USE POWEBI DESKTOP AND CREATE A DASHBOARD
* PUBLISH THE DASHBOARD INTO POWERBI SERVICE
* PUBLISH TO WEB THE DASHBOARD AND EMBED CODE TO THE HTML

8. RUN THE FLASK APPLICATION LOCALLY
python app.py

9. CREATE THE REQUIREMENTS FILE
pip freeze > requirements.txt

10. UPLOAD THE FLASK STRUCTURE TO A GITHUB REPOSITORY 
* UPLOAD FILES, FOLDERS AND THE REQUIREMENTS.TXT (ADDING IN THE END THE LIBRARY: Gunicorn)

DEPLOYMENT IN RENDER

10. CREATE A RENDER ACCOUNT (https://render.com/)

11. SELECT DEPLOY WEB SERVICE

12. LINK A GITHUB ACCOUNT

13. SELECT A GITHUB REPOSITORY AND CONNECT TO IT

14. WITHIN THE CONFIGURATION SELECT FREE INSTANCE TYPE AND AS ENVIRONMENT VARIABLE PUT: PYTHON_VERSION 3.11.2

15. CLICK ON DEPLOY WEB SERVICE AND WAIT UNTIL THE STATUS OF THE DEPLOY IS ON LIVE

16. ACCESS THROUGH THE LINK GIVEN BY RENDER TO THE WEB APPLICATION (https://big-data-visualization-ssg0.onrender.com)
