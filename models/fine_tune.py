from transformers import Trainer, TrainingArguments, AutoModelForCausalLM


def fine_tune_model(model_name, train_dataset, val_dataset, tokenizer, config, training_config):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, use_auth_token=True).to(config.DEVICE)

    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=training_config["learning_rate"],
        # Зменшення батчу
        per_device_train_batch_size=training_config["batch_size"],
        per_device_eval_batch_size=training_config["batch_size"],
        num_train_epochs=training_config["epochs"],
        weight_decay=0.01,
        logging_dir=config.LOGGING_DIR,
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        eval_steps=100,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,  # Забезпечення коректної обробки даних
    )

    trainer.train()

    return model
