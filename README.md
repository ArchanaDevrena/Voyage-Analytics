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
| Backend | Python 3.x, Flask (REST API) |
| Machine Learning | Scikit-learn |
| Model Tracking | MLflow |
| Database | PostgreSQL |
| Containerization | Docker, Docker Compose |
| Orchestration | Kubernetes |
| Deployment | Streamlit Cloud / Docker / Kubernetes |
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
- Flask REST API layer for modular backend access
- Responsive, mobile-friendly interface built with Streamlit
- PostgreSQL database for user and travel data persistence
- Containerized deployment via Docker and Docker Compose
- Kubernetes-ready with production deployment manifests
- Streamlit Cloud deployment for public accessibility

---

## Project Structure

```
Voyage-Analytics/
│
├── data/                                   # Raw datasets
│   ├── flights.csv                         # Historical flight pricing data
│   ├── hotels.csv                          # Hotel records and metadata
│   └── users.csv                           # User data for offline fallback
│
├── database/                               # Database utilities
│   ├── db.py                               # PostgreSQL connection and query logic
│   └── test_insert.py                      # Database insertion test script
│
├── flask_api/                              # REST API layer (Flask)
│   └── app.py                              # API routes and endpoint definitions
│
├── k8s/                                    # Kubernetes deployment configuration
│   └── deployment.yaml                     # K8s deployment and service manifest
│
├── models/                                 # Trained model artifacts
│   ├── flight/                             # Flight price prediction models
│   │   ├── flight_model.json               # Model architecture (JSON format)
│   │   ├── flight_model.pkl                # Serialized model (primary)
│   │   ├── flight_model_depth5_lr0.2.json  # Tuned model variant (JSON)
│   │   └── flight_model_depth5_lr0.2.pkl   # Tuned model variant (serialized)
│   └── recommendation/                     # Recommendation engine artifacts
│       ├── complete_data.joblib            # Full dataset for inference
│       ├── hotel_features.joblib           # Encoded hotel feature matrix
│       ├── hotel_similarity.joblib         # Item-based similarity matrix
│       ├── user_hotel_matrix.joblib        # User-hotel interaction matrix
│       ├── user_similarity.joblib          # User-based similarity matrix
│       └── users_data.joblib               # User profile data
│
├── notebooks/                              # Jupyter development notebooks
│   ├── flight_price_model.ipynb            # Flight model training and evaluation
│   └── Recommendation system.ipynb         # Recommendation engine development
│
├── templates/                              # HTML templates (Flask frontend)
│
├── flight_price_model.py                   # Flight model training script
├── inference.py                            # Real-time model inference module
├── app.py                                  # Streamlit application (main entry point)
├── Dockerfile                              # Docker container configuration
├── docker-compose.yml                      # Multi-service Docker orchestration
├── README.md                               # Project documentation
└── .gitignore                              # Git ignore rules
```

---

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- PostgreSQL (required for user management and booking data)
- Docker and Docker Compose (for containerized deployment)
- kubectl (for Kubernetes deployment, optional)
- Git

### Step 1 — Clone the Repository

```bash
git clone https://github.com/VoyageAnalytics/Voyage-Analytics.git
cd Voyage-Analytics
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

If no database is configured, the application falls back to CSV-based user data from the `data/` directory.

---

## Running the Application

### Option 1 — Streamlit (Local Development)

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Option 2 — Flask API

```bash
cd flask_api
python app.py
```

The REST API will be available at `http://localhost:5000`

### Option 3 — Docker Compose (Recommended)

```bash
docker-compose up --build
```

This starts all services (Streamlit frontend, Flask API, and database) in coordinated containers.

### Option 4 — Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
```

Refer to the `k8s/` directory for service and ingress configuration details.

### Default Credentials (Demo Users)

All pre-loaded dataset users share the following default password:

```
Password: password123
```

Users are identified by a numeric user code starting from User 0. New users can register directly through the application interface.

---

## Deployment

The platform supports multiple deployment targets:

**Streamlit Cloud**
Connect the repository to Streamlit Cloud via the dashboard, configure secrets (database URL) through the secrets manager, and the app deploys automatically on each push to the main branch.

**Docker**
The included `Dockerfile` and `docker-compose.yml` enable containerized deployment to any Docker-compatible environment including AWS ECS, Google Cloud Run, and Azure Container Instances.

**Kubernetes**
The `k8s/deployment.yaml` manifest defines the deployment, service, and resource configuration for production-grade orchestration via Kubernetes. Apply directly with `kubectl` or integrate into a CI/CD pipeline.

---

## Problem Statement

Flight ticket pricing is inherently dynamic and opaque. Prices shift based on factors including booking lead time, route demand, seat availability, and seasonal patterns. Most travelers book reactively rather than strategically, often paying more than necessary.

Voyage Analytics tackles this problem by applying supervised machine learning to historical flight data, enabling the system to learn pricing patterns and generate accurate predictions for user-specified routes and dates. Combined with a recommendation engine that personalizes hotel suggestions based on user demographics and past behavior, the platform transforms raw travel data into practical, decision-ready intelligence.

---

## Conclusion

Voyage Analytics represents a complete, production-oriented travel intelligence platform that brings together machine learning, data analytics, and modern web deployment in a unified system. The platform is built to be extensible — new models can be versioned and deployed through MLflow, new data sources can be integrated into the pipeline, and the recommendation engine can be refined as user data grows.

The system demonstrates how data science and engineering can be combined to create practical tools that empower users with information they can act on, reducing guesswork and improving decision-making in the context of travel planning.
