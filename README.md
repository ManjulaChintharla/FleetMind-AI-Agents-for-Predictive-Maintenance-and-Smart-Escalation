## **Overview**:

The Fleet Maintenance AI Scenario is an end-to-end AI solution designed to predict vehicle maintenance needs using real-time telemetry data. It leverages the full Microsoft Azure ecosystem—Azure Data Factory, PostgreSQL, Azure Machine Learning, Azure AI Search, and Azure OpenAI—to provide predictive insights and intelligent recommendations for the transportation and logistics sector.

## Problem Statement

Fleet managers lack a proactive system to anticipate vehicle failures and schedule maintenance efficiently, leading to increased downtime and operational costs.

## Solution Overview

- Trained a predictive model (ExtraTreesClassifier) using Azure Machine Learning

- Ingested and stored cleaned telemetry data in Azure PostgreSQL

- Generated semantic embeddings for maintenance documents using Azure OpenAI

- Created vector and keyword indexes using Azure Cognitive Search

- Built a GPT-powered chat interface to retrieve predictions and document suggestions
  
# Azure End-to-End AI Fleet Maintenance Platform

This project is a full-stack AI-powered pipeline tailored for data scientists and MLOps engineers. It shows how to build a production-ready AI solution—from ingesting raw IoT fleet data to delivering intelligent predictions and document-based insights via a chat interface. The platform integrates real-time sensor analysis, machine learning, vector-based search, and conversational AI—all within Microsoft Azure.

###  Pipeline Overview

- **Azure Data Factory** – Data ingestion from multiple sources

- **Azure PostgreSQL** – Centralized storage for cleaned fleet data

- **Azure Machine Learning** – Model training and deployment

- **Azure OpenAI Embeddings + PostgreSQL** – Text vectorization and document storage

- **Azure AI Search** – Vector and keyword search for maintenance documentation

- **Azure OpenAI (GPT+ textembedding)** – Chatbot interface that merges predictions with document insights

###  Features

- Predict upcoming vehicle maintenance using ML

- Search maintenance documents with semantic understanding

- Interact via a chatbot powered by Azure OpenAI

- Real-time connection between chat inputs, prediction models, and search results


##  How to Run
- run script.sh to setup environment
  
- Set up your environment variables in a .env file

- Run the notebooks sequentially from the notebooks/ directory

- Deploy and interact with the chatbot for live predictions and document tips


