import logging
from transformers import GPT2LMHeadModel, TrainingArguments, Trainer
from data.dataset_prep import prepare_dataset

logging.basicConfig(level=logging.DEBUG)


def fine_tune_model(model_name, train_dataset, val_dataset, tokenizer, learning_rate, batch_size, epochs):
    model = GPT2LMHeadModel.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=50,
        save_total_limit=2,
        learning_rate=learning_rate,
        report_to="none",  # Avoid logging to external services
    )

    # Using default data collator to handle padding correctly
    from transformers import default_data_collator
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train()

    return model


def main():
    model_name = "gpt2"  # You can change this to another model
    logging.info("Preparing dataset...")
    train_dataset, val_dataset, test_dataset, tokenizer = prepare_dataset(
        model_name)

    logging.debug(f"Input IDs Shape: {train_dataset[0]['input_ids'].shape}, "
                  f"Labels Shape: {train_dataset[0]['labels'].shape}")

    experiment = {
        "learning_rate": 5e-5,
        "batch_size": 4,
        "epochs": 3
    }

    logging.info(f"Running Experiment 1: {experiment}")
    fine_tune_model(
        model_name,
        train_dataset,
        val_dataset,
        tokenizer,
        learning_rate=experiment["learning_rate"],
        batch_size=experiment["batch_size"],
        epochs=experiment["epochs"]
    )


if __name__ == "__main__":
    main()
