import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification


def get_model_and_tokenizer(checkpoint=None):
    
    model = DistilBertForSequenceClassification.from_pretrained(
                'distilbert-base-uncased',
                num_labels=6
            )
    
    if checkpoint:
        print(f"Loading checkpoint: {checkpoint}")
        model.load_state_dict(torch.load(checkpoint, weights_only=True))

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    return model, tokenizer