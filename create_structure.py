import os

# Project structure as nested dictionary
structure = {
    "data": {
        "raw": {},
        "processed": {},
        "splits": {}
    },
    "models": {
        "checkpoints": {},
        "exports": {}
    },
    "triton_model_repo": {
        "crop_disease_classifier": {
            "1": {},
            "config.pbtxt": None
        }
    },
    "notebooks": {},
    "logs": {},
    "docker": {},
    "tests": {
        "test_utils.py": None
    },
    "src": {
        "__init__.py": None,
        "__main__.py": None,
        "cli.py": None,
        "utils": {
            "__init__.py": None,
            "logger.py": None,
            "exception.py": None,
            "config.py": None,
            "helpers.py": None
        },
        "ingestion": {
            "dataset_downloader.py": None,
            "drone_stream.py": None,
            "api_ingest.py": None
        },
        "preprocessing": {
            "transforms.py": None,
            "augmentation.py": None
        },
        "datasets": {
            "leaf_dataset.py": None,
            "yield_dataset.py": None
        },
        "training": {
            "train.py": None,
            "loss_functions.py": None,
            "scheduler.py": None,
            "grad_cam.py": None
        },
        "models": {
            "__init__.py": None,
            "classifier.py": None,
            "detector.py": None,
            "yield_predictor.py": None
        },
        "evaluation": {
            "metrics.py": None,
            "evaluation.py": None,
            "visualization.py": None
        },
        "inference": {
            "local_infer.py": None,
            "triton_client.py": None,
            "postprocess.py": None
        },
        "yield_predictor": {
            "feature_engineering.py": None,
            "train_yield.py": None,
            "predict_yield.py": None
        },
        "api": {
            "app.py": None,
            "routes": {
                "infer.py": None,
                "yield.py": None,
                "alerts.py": None
            },
            "websocket.py": None
        },
        "dashboard": {
            "static": {},
            "src": {}
        },
        "monitoring": {
            "prometheus.py": None,
            "grafana_config.py": None,
            "health_checks.py": None
        }
    }
}

# Root-level files
root_files = [
    "setup.py",
    "requirements.txt",
    "config.yaml",
    "README.md",
    ".gitignore"
]

def create_structure(base_path, struct):
    for name, content in struct.items():
        path = os.path.join(base_path, name)
        if content is None:  # It's a file
            if not os.path.exists(path):
                with open(path, "w") as f:
                    pass  # create empty file
        else:  # It's a directory
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)

if __name__ == "__main__":
    base_dir = "."  # already in crop_ai/
    
    # Create root-level files
    for file in root_files:
        if not os.path.exists(file):
            with open(file, "w") as f:
                pass
    
    # Create nested structure
    create_structure(base_dir, structure)
    
    print("✅ Project structure created successfully!")
