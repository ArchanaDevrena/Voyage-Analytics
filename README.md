# Voyage Analytics
### Travel Intelligence and Prediction Platform

---

## Overview

Voyage Analytics is a data-driven travel intelligence platform that integrates machine learning with an interactive web interface to deliver actionable travel insights. The platform enables users to predict flight prices in real time, receive personalized hotel recommendations, and explore analytical dashboards — all powered by trained ML models and a clean, responsive UI built with Streamlit.

The system is designed with scalability and modularity in mind, incorporating MLflow for experiment tracking, PostgreSQL for user management, and a collaborative filtering engine for recommendation generation.

---

## Table of Contents

1. [Project Objectives](#project-objectives)
2. [Technology Stack](#technology-stack)
3. [System Architecture](#system-architecture)
4. [Machine Learning Pipeline](#machine-learning-pipeline)
5. [System Modules](#system-modules)
6. [Key Features](#key-features)
7. [Project Structure](#project-structure)
8. [Setup and Installation](#setup-and-installation)
9. [Running the Application](#running-the-application)
10. [Deployment](#deployment)
11. [Problem Statement](#problem-statement)
12. [Conclusion](#conclusion)

---

## Project Objectives

Flight ticket prices fluctuate continuously due to demand elasticity, seasonal variation, airline pricing strategies, and broader market dynamics. Travelers frequently lack the analytical tools to make informed booking decisions.

Voyage Analytics addresses this by:

- Predicting flight prices using a trained machine learning model
- Providing intelligent, user-specific travel recommendations
- Tracking model performance and experiment history using MLflow
- Delivering an interactive, user-friendly interface accessible via web browser

---

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | Python 3.x |
| Machine Learning | Scikit-learn |
| Model Tracking | MLflow |
| Database | PostgreSQL / CSV |
| Deployment | Streamlit Cloud |
| Version Control | Git and GitHub |

---

## System Architecture

The platform is divided into three primary layers:

**Presentation Layer**
The Streamlit-based frontend handles user interaction, form input, and visualization rendering. It communicates with the backend logic to display predictions, recommendations, and analytics dashboards.

**Application Layer**
The Python backend manages authentication, request routing, data preprocessing, and model inference. It interfaces with both the database and the trained ML model.

**Data Layer**
User data and travel records are stored in a PostgreSQL database. Model artifacts are versioned and stored via MLflow. Raw travel data is maintained as structured CSV files.

---

## Machine Learning Pipeline

The ML pipeline processes historical flight data through the following stages:

1. Data Collection — Aggregation of flight records including route, timing, and pricing
2. Data Cleaning — Handling of missing values, outliers, and format inconsistencies
3. Feature Engineering — Construction of derived features including distance buckets, time-of-week indicators, and seasonal flags
4. Model Training — Regression model trained using Scikit-learn on historical pricing data
5. Model Evaluation — Cross-validation and performance metrics including MAE and RMSE
6. Experiment Tracking — All training runs logged to MLflow with parameters, metrics, and artifacts
7. Model Serialization — Final model saved as a `.pkl` file for inference
8. Deployment — Model served in real time via Streamlit using the `inference.py` module

**Prediction Features**

The model predicts flight prices based on the following inputs:

- Origin city
- Destination city
- Airline / agency
- Flight distance (km)
- Flight duration (hours)
- Travel date
- Flight type (economy, business, etc.)

---

## System Modules

### User Module

The User module is designed for travelers and provides the following capabilities:

- Secure registration and login
- Real-time flight price prediction
- Personalized hotel recommendations
- Travel insights and analytics dashboard
- User profile management

Authentication is handled securely with hashed credentials stored in the database. Existing dataset users are pre-loaded with a default password for demonstration purposes.

### Admin Module

The Admin module provides system administrators with tools to manage and monitor the platform:

- View and manage registered user records
- Monitor prediction usage and system activity
- Manage and update travel datasets
- Track MLflow experiment history
- Update or replace deployed model versions

### Prediction and Recommendation Module

This module is the analytical core of the platform:

- **Flight Price Prediction** — Real-time inference using the trained ML model
- **Hotel Recommendations** — Hybrid collaborative filtering using user similarity and item-based methods
- **Data Preprocessing** — Automated pipeline for feature transformation and encoding
- **Adaptive Scoring** — Recommendation scores adjusted based on user demographics and booking history
- **MLflow Logging** — All inference events and experiments tracked for audit and improvement

---

## Key Features

- Real-time flight price prediction powered by a trained regression model
- Personalized hotel recommendation engine using hybrid collaborative filtering
- Interactive travel analytics dashboard with demographic and booking insights
- Secure user registration and login with session management
- MLflow integration for experiment tracking and model versioning
- Responsive, mobile-friendly interface built with Streamlit
- PostgreSQL database for user and travel data persistence
- Streamlit Cloud deployment for public accessibility

---

## Project Structure

```
Voyage-Analytics/
│
├── app.py                        # Main Streamlit application entry point
├── inference.py                  # Model inference logic
├── recommendation_engine.py      # Collaborative filtering recommendation module
│
├── models/
│   ├── flight_model.pkl          # Serialized trained ML model
│   └── recommendation/           # Saved recommendation model artifacts
│       ├── user_hotel_matrix.pkl
│       ├── user_similarity.pkl
│       ├── hotel_similarity.pkl
│       ├── hotel_features.pkl
│       ├── complete_data.pkl
│       └── users_data.pkl
│
├── data/
│   └── flights.csv               # Historical flight dataset
│
├── database/
│   └── db.py                     # Database connection and query utilities
│
├── mlruns/                       # MLflow experiment tracking directory
│
├── screenshots/                  # UI screenshots for documentation
│
├── requirements.txt              # Python dependency list
└── README.md                     # Project documentation
```

---

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- PostgreSQL (optional, for full database functionality)
- Git

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/voyage-analytics.git
cd voyage-analytics
```

### Step 2 — Create a Virtual Environment

```bash
python -m venv venv

# Activate on macOS / Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure Environment Variables

Create a `.streamlit/secrets.toml` file with the following structure:

```toml
DATABASE_URL = "postgresql://username:password@host:port/dbname"
```

If no database is configured, the application falls back to CSV-based user data.

---

## Running the Application

### Start MLflow Tracking Server (Optional)

```bash
mlflow ui
```

MLflow UI will be accessible at `http://localhost:5000`

### Launch the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## Deployment

The application is deployed on **Streamlit Cloud** and is publicly accessible via a hosted URL. The deployment process involves:

1. Pushing the repository to GitHub
2. Connecting the repository to Streamlit Cloud via the dashboard
3. Configuring secrets (database URL) through the Streamlit Cloud secrets manager
4. Automatic deployment on each push to the main branch

MLflow experiment tracking is maintained locally or on a separate tracking server, independent of the Streamlit deployment.

---

## Problem Statement

Flight ticket pricing is inherently dynamic and opaque. Prices shift based on factors including booking lead time, route demand, seat availability, and seasonal patterns. Most travelers book reactively rather than strategically, often paying more than necessary.

Voyage Analytics tackles this problem by applying supervised machine learning to historical flight data, enabling the system to learn pricing patterns and generate accurate predictions for user-specified routes and dates. Combined with a recommendation engine that personalizes hotel suggestions based on user demographics and past behavior, the platform transforms raw travel data into practical, decision-ready intelligence.

---

## Conclusion

Voyage Analytics represents a complete, production-oriented travel intelligence platform that brings together machine learning, data analytics, and modern web deployment in a unified system. The platform is built to be extensible — new models can be versioned and deployed through MLflow, new data sources can be integrated into the pipeline, and the recommendation engine can be refined as user data grows.

The system demonstrates how data science and engineering can be combined to create practical tools that empower users with information they can act on, reducing guesswork and improving decision-making in the context of travel planning.

---

*For issues, contributions, or questions, please open a GitHub issue or submit a pull request.*
