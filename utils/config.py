import torch

MODEL_NAME = "distilgpt2"  # Модель для налаштування #gpt2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Шляхи для збереження результатів
OUTPUT_DIR = "./output/models"
LOGGING_DIR = "./output/logs"
PLOTS_DIR = "./output/plots"

# Набір експериментів
EXPERIMENTS = [
    {"learning_rate": 5e-5, "batch_size": 4, "epochs": 3},
    {"learning_rate": 3e-5, "batch_size": 8, "epochs": 5},
]
