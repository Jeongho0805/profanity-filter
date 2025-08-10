
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

tokenizer = None

def preprocess_function(examples):
    tokenized_inputs = tokenizer(examples["text"], padding="max_length", truncation=True)

    labels = []
    for label_list in examples["label"]:
        if label_list == [8]:
            labels.append(0)
        else:
            labels.append(1)
    tokenized_inputs["label"] = labels

    return tokenized_inputs

def main():
    global tokenizer
    model_name = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    dataset = load_dataset("jeanlee/kmhas_korean_hate_speech")
    dataset = dataset.filter(lambda x: isinstance(x['text'], str) and len(x['text']) > 0)
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=['text'])

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    print(f"훈련 데이터 크기: {len(train_dataset)}")
    print(f"검증 데이터 크기: {len(eval_dataset)}")


    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=3e-5,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=500,
        save_steps=1000,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    model.save_pretrained("./chat-profanity-filter")
    tokenizer.save_pretrained("./chat-profanity-filter")

if __name__ == "__main__":
    main()
