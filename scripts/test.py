from transformers import AutoTokenizer, AutoModelForCausalLM
from data.dataset_prep import prepare_dataset
from models.evaluate import evaluate_model
from utils import config


def main():
    # Завантаження токенізатора та моделі
    tokenizer = AutoTokenizer.from_pretrained(config.FINE_TUNED_MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        config.FINE_TUNED_MODEL_DIR).to(config.DEVICE)

    # Підготовка даних
    _, test_dataset = prepare_dataset(config.MODEL_NAME)

    # Оцінка моделі
    evaluate_model(model, test_dataset, tokenizer)


if __name__ == "__main__":
    main()
