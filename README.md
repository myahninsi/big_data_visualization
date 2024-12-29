# Big Data Visualization Project

## Overview
This project showcases an end-to-end implementation of a Flask-based web application for data visualization and prediction. The application integrates machine learning models, Python libraries, and a Power BI dashboard to provide an interactive data analysis experience.

## Problem Statement
E-commerce platforms face challenges in understanding customer behaviors, predicting key metrics like sales, and providing actionable insights to decision-makers. This project addresses these gaps by integrating machine learning, visualization, and a user-friendly interface to enhance data-driven decision-making.

## Features
- **Flask Application**: Handles web requests, model predictions, and serves the HTML templates.
- **Machine Learning**: Includes a pre-trained model for predictions.
- **Interactive Dashboard**: Embedded Power BI dashboard for visualizing data insights.
- **Deployment**: Hosted on Render with a live link.

## Project Workflow

### 1. Problem Definition
   - Identify key challenges in customer behavior analysis and visualization for actionable insights.
   - Define project goals: predictive modeling, interactive visualization, and deployment.

### 2. Data Collection and Preprocessing
   - Collect dataset from sources like Kaggle.
   - Perform data cleaning, including handling missing values and outliers.
   - Conduct feature engineering to prepare high-quality inputs for modeling.

### 3. Exploratory Data Analysis (EDA)
   - Visualize data distributions and relationships using Python libraries like Matplotlib and Seaborn.
   - Uncover insights such as sales trends, customer behaviors, and feature importance.

### 4. Model Development
   - Implement multiple regression models: Linear, Lasso, Ridge, and ElasticNet Regression.
   - Evaluate models using metrics such as R² and RMSE to ensure accuracy.

### 5. Visualization and Dashboard Creation
   - Create an interactive Power BI dashboard to present insights.
   - Embed the Power BI dashboard into the Flask application for seamless user access.

### 6. Application Development
   - Develop a Flask-based front-end to enable real-time interaction with predictive models.
   - Structure the application with modular components: `app.py`, `predictor.py`, and organized folders for templates, static files, and data.

### 7. Deployment
   - Generate a `requirements.txt` file using `pip freeze`.
   - Host the application on Render with environment configuration (e.g., Python version).
   - Test the live application for functionality and accessibility.

## Environment Setup
1. **Create a Virtual Environment**:
   ```bash
   py -3 -m venv .venv
   ```
2. **Activate the Virtual Environment**:
   ```bash
   .venv\Scripts\activate
   ```
3. **Define Python Version**: Python 3.11.2

## Dependencies
Install required libraries:
```bash
pip install Flask==3.0.3
pip install pandas==2.2.2
pip install matplotlib==3.9.1.post1
pip install seaborn==0.13.2
pip install numpy==2.0.1
pip install scikit-learn==1.4.2
pip install scipy==1.14.0
pip install openpyxl==3.1.5
```

## Flask Application Structure
- **`app.py`**: Manages Flask and handles web requests.
- **`predictor.py`**: Manages the machine learning model and returns predictions.
- **Folders**:
  - **`templates`**: Contains HTML files.
  - **`static`**: Contains CSS files and images.
  - **`pickle`**: Stores the trained model in pickle format.
  - **`data`**: Stores necessary Excel or CSV files.

## Power BI Dashboard Integration
1. Create a dashboard using Power BI Desktop.
2. Publish the dashboard to Power BI Service.
3. Generate an embed code and include it in the HTML template.

## Running Locally
1. Start the Flask application:
   ```bash
   python app.py
   ```
2. Access the application at `http://localhost:5000`.

## Deployment to Render
1. Create a [Render](https://render.com/) account.
2. Deploy as a web service:
   - Connect your GitHub repository.
   - Select the free instance type.
   - Add an environment variable: `PYTHON_VERSION=3.11.2`.
3. Deploy and access your web application via the provided link.
   - Example: [https://big-data-visualization-ssg0.onrender.com](https://big-data-visualization-ssg0.onrender.com)

## Folder Structure
```
.
├── app.py
├── predictor.py
├── templates/
├── static/
├── pickle/
├── data/
├── requirements.txt
```

## Live Demo
Access the live application [here](https://big-data-visualization-ssg0.onrender.com).

## Skills Highlighted
- Machine Learning
- Regression Models
- Data Visualization
- Python
- Power BI
- Flask
- GitHub
- Cloud Deployment