from evaluate import load as load_metric


def evaluate_model(model, test_dataset, tokenizer, config):
    bleu_metric = load_metric("bleu")
    rouge_metric = load_metric("rouge")

    references = []
    predictions = []

    for sample in test_dataset:
        input_ids = sample["input_ids"].unsqueeze(0).to(config.DEVICE)
        reference = tokenizer.decode(
            sample["labels"], skip_special_tokens=True)
        output_ids = model.generate(
            input_ids, max_length=50, num_beams=5, early_stopping=True)
        prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        references.append(reference)
        predictions.append(prediction)

    bleu_result = bleu_metric.compute(predictions=[pred.split() for pred in predictions],
                                      references=[[ref.split()] for ref in references])
    rouge_result = rouge_metric.compute(
        predictions=predictions, references=references)

    print(f"BLEU Score: {bleu_result['bleu']}")
    print(f"ROUGE Score: {rouge_result}")

    return bleu_result, rouge_result
