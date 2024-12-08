from transformers import AutoTokenizer
from datasets import load_dataset


def prepare_dataset(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Додавання pad_token, якщо його немає
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Завантаження набору даних
    dataset = load_dataset("imdb")

    # Розділення даних
    train_val_split = dataset["train"].train_test_split(test_size=0.2)
    train_split = train_val_split["train"]
    val_test_split = train_val_split["test"].train_test_split(test_size=0.5)
    val_split = val_test_split["train"]
    test_split = val_test_split["test"]

    def tokenize_function(examples):
        tokens = tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=512
        )
        # Додавання міток (labels) з ігноруванням `pad_token_id`
        tokens["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in input_ids]
            for input_ids in tokens["input_ids"]
        ]
        return tokens

    # Токенізація наборів
    train_dataset = train_split.map(
        tokenize_function, batched=True, remove_columns=["text"])
    val_dataset = val_split.map(
        tokenize_function, batched=True, remove_columns=["text"])
    test_dataset = test_split.map(
        tokenize_function, batched=True, remove_columns=["text"])

    # Налаштування форматування
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    test_dataset.set_format("torch")

    return train_dataset, val_dataset, test_dataset, tokenizer
