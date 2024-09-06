import os
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Load the pre-trained model and modify it for multi-class classification
num_labels = 5  # Example: Assume there are 5 different bug types
model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=num_labels)

# Example dataset preparation
def load_pytracebugs(dataset_path):
    buggy_code = []
    fixed_code = []
    labels = []
    for project in os.listdir(dataset_path):
        project_path = os.path.join(dataset_path, project)
        if os.path.isdir(project_path):
            for bug in os.listdir(project_path):
                bug_path = os.path.join(project_path, bug)
                if os.path.isdir(bug_path):
                    buggy_file = os.path.join(bug_path, 'buggy_version.py')
                    fixed_file = os.path.join(bug_path, 'fixed_version.py')
                    label_file = os.path.join(bug_path, 'label.txt')
                    if os.path.exists(buggy_file) and os.path.exists(fixed_file) and os.path.exists(label_file):
                        with open(buggy_file, 'r') as bf, open(fixed_file, 'r') as ff, open(label_file, 'r') as lf:
                            buggy_code.append(bf.read())
                            fixed_code.append(ff.read())
                            labels.append(int(lf.read().strip()))
    return buggy_code, fixed_code, labels

# Load and preprocess the dataset
dataset_path = "path/to/PyTraceBugs/data"
buggy_code, fixed_code, labels = load_pytracebugs(dataset_path)

# Tokenize the dataset
inputs = tokenizer(buggy_code, padding=True, truncation=True, return_tensors="pt")
dataset = Dataset.from_dict({
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"],
    "labels": torch.tensor(labels)
})

# Compute class weights
class_weights = compute_class_weight('balanced', classes=[0, 1, 2, 3, 4], y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Custom Trainer to include weighted loss
class CustomTrainer(Trainer):
    def __init__(self, model, args, train_dataset=None, eval_dataset=None, tokenizer=None, data_collator=None, class_weights=None):
        super().__init__(model, args, train_dataset=train_dataset, eval_dataset=eval_dataset, tokenizer=tokenizer, data_collator=data_collator)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_total_limit=2,
)

# Initialize the Custom Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,  # Normally, you would use a separate validation set
    tokenizer=tokenizer,
    class_weights=class_weights
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)

# Save the model
model.save_pretrained('./fine-tuned-codebert')
tokenizer.save_pretrained('./fine-tuned-codebert')

def predict_bug_type(code_snippet):
    # Tokenize the input code snippet
    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, padding=True)
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    # Apply softmax to get probabilities
    probs = torch.softmax(logits, dim=-1)
    pred_label = torch.argmax(probs, dim=-1).item()
    # Map the prediction to human-readable labels
    labels = {0: 'SyntaxError', 1: 'LogicalError', 2: 'TypeError', 3: 'NameError', 4: 'AttributeError'}
    prediction = labels[pred_label]
    return prediction, probs[0][pred_label].item()

# Example usage
code_snippet = """
def add(a, b):
    return a + b
"""

prediction, confidence = predict_bug_type(code_snippet)
print(f"Prediction: {prediction}, Confidence: {confidence:.2f}")
