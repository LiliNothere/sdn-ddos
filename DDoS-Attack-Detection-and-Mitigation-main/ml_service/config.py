"""
ML Service Configuration
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Service settings
    service_host: str = "0.0.0.0"
    service_port: int = 8080
    grpc_port: int = 50051
    log_level: str = "INFO"

    # Model settings
    model_dir: str = "/app/models"
    default_model: str = "RandomForest"
    model_version: str = "1.0.0"

    # Training settings
    dataset_path: str = "/app/data/FlowStatsfile.csv"
    test_size: float = 0.25
    random_state: int = 42

    # Detection settings
    stats_poll_interval: int = 10  # seconds
    ddos_threshold_percent: float = 20.0  # Alert if >20% DDoS
    legitimate_threshold_percent: float = 80.0

    # Faucet integration
    faucet_host: str = "localhost"
    faucet_rest_port: int = 9302  # Faucet Prometheus port
    gauge_prometheus_port: int = 9303
    faucet_acl_name: str = "block-malicious"

    # Mitigation settings
    mitigation_duration: int = 120  # seconds
    mitigation_cooldown: int = 300  # seconds between mitigations

    # Prometheus metrics
    metrics_enabled: bool = True
    metrics_port: int = 9090

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


# Global settings instance
settings = get_settings()

# Model parameters for different algorithms
MODEL_PARAMS = {
    "RandomForest": {
        "n_estimators": 100,
        "criterion": "entropy",
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1
    },
    "KNN": {
        "n_neighbors": 5,
        "metric": "minkowski",
        "p": 2,
        "weights": "distance"
    },
    "DecisionTree": {
        "criterion": "entropy",
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42
    },
    "LogisticRegression": {
        "solver": "liblinear",
        "C": 1.0,
        "max_iter": 1000,
        "random_state": 42
    },
    "NaiveBayes": {
        "var_smoothing": 1e-9
    },
    "SVM": {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "random_state": 42,
        "probability": True
    }
}

# Feature columns (must match training data)
FEATURE_COLUMNS = [
    'timestamp', 'datapath_id', 'flow_id', 'ip_src', 'tp_src',
    'ip_dst', 'tp_dst', 'ip_proto', 'icmp_code', 'icmp_type',
    'flow_duration_sec', 'flow_duration_nsec', 'idle_timeout',
    'hard_timeout', 'flags', 'packet_count', 'byte_count',
    'packet_count_per_second', 'packet_count_per_nsecond',
    'byte_count_per_second', 'byte_count_per_nsecond'
]

# Columns to preprocess (remove dots from IPs)
IP_COLUMNS = ['ip_src', 'tp_src', 'ip_dst', 'tp_dst', 'ip_proto']

# Target column
TARGET_COLUMN = 'label'  # 0 = benign, 1 = ddos