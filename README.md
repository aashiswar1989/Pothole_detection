# 🕳️ Pothole Detection MLOps Pipeline 🚗  
An end-to-end Computer Vision pipeline for detecting road potholes using YOLOv8, integrated with MLflow, Docker, and AWS ECS for full MLOps deployment.

![Architecture Diagram](./assets/architecture.png)

## Overview

Road damage detection plays a vital role in smart city maintenance and driver safety.
This project implements a production-style MLOps pipeline for detecting potholes in road images, from data ingestion to real-time deployment.

The system automatically:
1. Fetches training data from an AWS S3 bucket
2. Trains a YOLOv8 model with experiment tracking via MLflow
3. Stores trained models and metrics back to S3
4. Deploys a FastAPI inference service and Gradio UI through a multi-container setup on AWS ECS
5. Automates builds and deployments using GitHub Actions CI/CD

This project demonstrates not just computer vision expertise, but also end-to-end MLOps implementation.


## 🧠 System Architecture

The project follows a modular, production-style MLOps design, separating **training**, **evaluation**, and **deployment** workflows for better scalability and maintenance.

### 🔹 Training & Experimentation Flow
- **Data Ingestion**: Fetches image dataset from AWS S3 using `boto3`.
- **Data Validation**: Ensures proper annotation formats and cleans invalid data.
- **Model Training**: Trains YOLOv8 model with hyperparameter tracking in **MLflow**.
- **Model Evaluation**: Evaluates model and logs metrics (mAP, precision, recall) in MLflow.
- **Artifacts Management**: Trained model, metrics, and reports are versioned and stored back to S3.

### 🔹 CI/CD & Deployment Flow
- **CI/CD (GitHub Actions)**:
  - Job 1: Linting and dependency setup.
  - Job 2: Builds multi-container Docker images (FastAPI + Gradio), pushes them to DockerHub/ECR.
- **Deployment (AWS ECS)**:
  - ECS Service runs two containers:
    - **FastAPI** – for serving inference API.
    - **Gradio** – for visual interface and user interaction.
  - ECS task uses images from ECR and exposes public IP for real-time access.

---

## ⚙️ Tech Stack

| Category | Tools / Services |
|-----------|------------------|
| **ML Framework** | YOLOv8 (Ultralytics) |
| **Experiment Tracking** | MLflow |
| **Cloud Storage** | AWS S3 |
| **Model Deployment** | FastAPI + Gradio |
| **Containerization** | Docker + Docker Compose |
| **Cloud Orchestration** | AWS ECS (Fargate) |
| **CI/CD** | GitHub Actions |
| **Language** | Python |
| **Environment Management** | requirements.txt + Dockerfile |
| **Version Control** | Git + GitHub |

---

## 🚀 Key Features

- Modular pipeline covering **data ingestion → validation → training → evaluation**.  
- **MLflow integration** for model metrics and experiment tracking.  
- **S3-based artifact management** for remote storage and reproducibility.  
- **FastAPI service** for scalable model inference.  
- **Gradio UI** for interactive result visualization.  
- **GitHub Actions CI/CD** automating build, lint, and deployment workflows.  
- **Multi-container architecture** deployed via **AWS ECS** with public endpoint.  
- Designed with **MLOps best practices** (traceability, reproducibility, deployment automation).
