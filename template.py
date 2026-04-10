import os
from pathlib import Path

path = Path(os.getcwd())
print(path)

file_struct = [
    f"{path}/artifacts/.gitkeep",
    f"{path}/logs/.gitkeep",

    f"{path}/notebook/lstm.ipynb",

    f"{path}/src/__init__.py",

    f"{path}/src/components/__init__.py",
    f"{path}/src/components/data_ingestion.py",
    f"{path}/src/components/data_transformation.py",
    f"{path}/src/components/model_trainer.py",

    f"{path}/src/pipeline/__init__.py",
    f"{path}/src/pipeline/training_pipeline.py",

    f"{path}/src/utils/__init__.py",
    f"{path}/src/utils/logger.py",
    f"{path}/src/utils/exception.py",

    f"{path}/src/app.py",

    f"{path}/dataset/.gitkeep",

    f"{path}/main.py",
    f"{path}/Dockerfile",
    f"{path}/requirements.txt"
]

for filepath in file_struct:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, "w") as f:
            pass
    else:
        print(f"File already exists: {filepath}")