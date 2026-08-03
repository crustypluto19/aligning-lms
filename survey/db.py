"""
MongoDB connection utilities for DP Text Evaluation Survey.

Supports both local MongoDB and MongoDB Atlas.
Connection URI is read from environment variables.
"""

import os
import streamlit as st
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

@st.cache_resource
def get_mongodb_client():
    """
    Establishes connection to MongoDB using Streamlit secrets, env var, or localhost.
    """
    mongodb_uri = None
    
    # Try to get from Streamlit secrets first
    try:
        if hasattr(st, 'secrets') and 'MONGODB_URI' in st.secrets:
            mongodb_uri = st.secrets['MONGODB_URI']
            print("📍 Using MongoDB URI from Streamlit secrets")
    except (ImportError, FileNotFoundError, KeyError):
        pass
    
    # Fall back to environment variable (for CLI scripts)
    if mongodb_uri is None:
        mongodb_uri = os.getenv("MONGODB_URI")
        if mongodb_uri:
            print("📍 Using MongoDB URI from environment variable")
    
    # Final fallback to localhost
    if mongodb_uri is None:
        mongodb_uri = "mongodb://localhost:27017/"
        print("📍 Using default MongoDB URI (localhost)")
    
    try:
        client = MongoClient(
            mongodb_uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            socketTimeoutMS=10000
        )
        
        # Verify connection
        client.admin.command('ping')
        
        return client
    
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        raise ConnectionFailure(f"Failed to connect to MongoDB at {mongodb_uri[:50]}...: {str(e)}")


@st.cache_resource
def get_database(db_name=st.secrets.get("MONGODB_DB_NAME", "dp_survey_dev")):
    """Returns the MongoDB database instance."""
    client = get_mongodb_client()
    return client[db_name]


def get_collections(db):
    """Returns questions, responses, and respondents collections."""
    questions = db["questions"]
    responses = db["responses"]
    respondents = db["respondents"]
    
    return questions, responses, respondents


def create_indexes(db):
    """Creates indexes for querying variant_id, dataset, answered_count, and (method, epsilon)."""
    questions = db["questions"]
    
    questions.create_index("variant_id", unique=True)
    questions.create_index("dataset")
    questions.create_index("answered_count")
    questions.create_index("reference_id")
    
    questions.create_index([("method", 1), ("epsilon", 1)])
    
    responses = db["responses"]
    responses.create_index("user_id")
    responses.create_index("timestamp")
    
    respondents = db["respondents"]
    respondents.create_index("user_id", unique=True)
    
    print("Indexes created successfully")


def reset_database(db):
    """
    Drops all collections in the database.
    Warning: This deletes all data.
    """
    db["questions"].drop()
    db["responses"].drop()
    db["respondents"].drop()
    print("Database reset complete")
