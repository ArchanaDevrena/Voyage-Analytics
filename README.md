# ✈️ Voyage Analytics
Travel Intelligence & Prediction Platform

Voyage Analytics is a smart travel intelligence web application designed to analyze travel data, predict flight prices, and provide personalized travel recommendations. The platform integrates Machine Learning models with an interactive web interface to deliver data-driven travel insights.

## 📌 Project Overview

The basic concept of this project Voyage Analytics is to provide an intelligent travel analytics platform that helps users predict flight prices, get personalized recommendations, and analyze travel trends using Machine Learning.

The system integrates ML-based flight price prediction, user management, and interactive dashboards. The application has been deployed using Streamlit and includes MLflow integration for experiment tracking and model management.

## 🛠 Tools and Technologies

Frontend : Streamlit
Backend : Python
Machine Learning : Scikit-learn
Model Tracking : MLflow
Database : CSV / Database Integration (users & travel data)
Deployment : Streamlit Cloud / Local Streamlit Server
Version Control : Git & GitHub

## 🤖 Machine Learning Integration

The system includes:

Flight Price Prediction Model

MLflow experiment tracking

Model versioning

Saved .pkl trained model files

Real-time inference using trained model

Feature engineering pipeline

Data preprocessing and transformation

The ML model predicts flight prices based on:

Source

Destination

Airline

Distance

Duration

Travel Date

Flight Type

## 🧩 System Modules

The system has three main modules:

User

Admin

Prediction & Recommendation Engine

## 👤 User Module

The User module is designed for travelers who want to:

Register and login securely

Predict flight prices

Get travel recommendations

View travel insights

Access personalized suggestions

Users can enter flight details such as source, destination, distance, duration, and airline to get real-time price predictions powered by the ML model.

The module securely stores user details and manages authentication.

## 🛠 Admin Module

The Admin module is designed for system administrators who manage:

User records

Travel data

Model updates

System monitoring

Admins can:

View registered users

Monitor prediction usage

Manage datasets

Track MLflow experiments

Update models when required

## 🧠 Prediction & Recommendation Module

This module handles:

Flight price prediction using trained ML model

Personalized travel recommendations

Data preprocessing pipeline

Feature transformation

Model inference

MLflow experiment logging

It ensures accurate and optimized price predictions based on historical travel data.

## 🚀 Features

✈️ Flight Price Prediction

📊 Travel Analytics Dashboard

🤖 MLflow Integration

🔐 Secure Login & Registration

📱 Mobile Screen Friendly Interface

🧠 Intelligent Recommendation System

📈 Model Experiment Tracking

☁️ Streamlit Deployment

📸 UI Screenshots Included

## 🏗 ML Pipeline

Data Collection

Data Cleaning

Feature Engineering

Model Training

Model Evaluation

MLflow Experiment Tracking

Model Saving (.pkl file)

Deployment with Streamlit

Real-time Prediction

## 🌐 Deployment

The application is deployed using:

Streamlit

MLflow for experiment tracking

GitHub for version control

The deployed app allows users to:

Register/Login

Predict flight prices

Get recommendations

Interact with dashboards

## ▶️ How to Run the Project
### 1️⃣ Clone the Repository
git clone https://github.com/your-username/voyage-analytics.git
cd voyage-analytics

### 2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

### 3️⃣ Install Dependencies
pip install -r requirements.txt

### 4️⃣ Run MLflow (Optional – for tracking)
mlflow ui


Then open:

http://localhost:5000

### 5️⃣ Run Streamlit App
streamlit run app.py

### 📁 Project Structure
Voyage-Analytics/
│
├── app.py
├── inference.py
├── database/
│   ├── db.py
│   └── users.csv
├── models/
│   └── flight_model.pkl
├── data/
├── screenshots/
├── requirements.txt
└── README.md

### 📸 Screenshots

Login & Register Page

Flight Prediction Page

Recommendation Dashboard

MLflow Tracking UI

Streamlit Deployment View

(Screenshots included in the project folder)

### 🎯 Problem Statement

Flight ticket prices fluctuate frequently due to demand, seasonality, airline pricing strategies, and travel patterns. Travelers often struggle to determine the best time to book flights.

Voyage Analytics solves this problem by:

Predicting flight prices using Machine Learning

Providing intelligent travel recommendations

Tracking model performance using MLflow

Delivering an interactive and user-friendly web interface

### 🌟 Conclusion

Voyage Analytics is a complete Travel Intelligence Platform that combines:

Machine Learning

Data Analytics

Web Deployment

Model Tracking

User Authentication

The system empowers travelers with accurate price predictions and smart travel insights while maintaining scalable and production-ready architecture.