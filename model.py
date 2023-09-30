from transformers import AutoModel, AutoTokenizer
import torch
from torch import nn

class SentimentClassifierWithMultipleHeads(nn.Module):
    def __init__(self, model_name, num_labels):
        super(SentimentClassifierWithMultipleHeads, self).__init__()
        self.model = AutoModel.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.num_labels = num_labels
        # 12 heads for BERT
        self.classification_heads = {f'head_{i}': torch.nn.Sequential(
            torch.nn.Linear(768, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, self.num_labels)) for i in range(12)}
        
        

    def forward(self, input_ids, attention_mask, labels=None):
        # get hidden states from each layer of BERT
        outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True, output_attentions=True)
        hidden_states = outputs[2]
        print(len(hidden_states), hidden_states[0].shape)
        # Pool outputs of all 128 tokens in each sequence per layer
        hidden_states = [torch.mean(layer, dim=1) for layer in hidden_states]
        print(len(hidden_states), hidden_states[0].shape)
        # pass each layer to its own classification head
        logits = {f'logits_{i}': self.classification_heads[f'head_{i}'](hidden_states[i+1]) for i in range(12)}
        print(len(logits), logits['logits_0'].shape)
        # Take softmax of logits to get probabilities
        probs = {f'probs_{i}': torch.nn.functional.softmax(logits[f'logits_{i}'], dim=-1) for i in range(12)}
        print(len(probs), probs['probs_0'].shape)
        # Compute loss for each head
        loss = {f'loss_{i}': torch.nn.functional.cross_entropy(logits[f'logits_{i}'], labels) for i in range(12)}
        print(len(loss), loss['loss_0'].shape)
        return loss, logits
    
    def predict(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs[2]
        hidden_states = [torch.mean(layer, dim=1) for layer in hidden_states]
        logits = {f'logits_{i}': self.classification_heads[f'head_{i}'](hidden_states[i+1]) for i in range(12)}
        probs = {f'probs_{i}': torch.nn.functional.softmax(logits[f'logits_{i}'], dim=-1) for i in range(12)}
        predictions = {f'predictions_{i}': torch.argmax(probs[f'probs_{i}'], dim=-1) for i in range(12)}
        return predictions
    
if __name__ == '__main__':
    model_name = 'bert-base-uncased'
    num_labels = 2
    model = SentimentClassifierWithMultipleHeads(model_name, num_labels)
    input_ids = torch.randint(0, 1000, (32, 128))
    attention_mask = torch.ones((32, 128))
    labels = torch.randint(0, 2, (32,))
    loss, logits = model(input_ids, attention_mask, labels)
    print(len(loss), loss['loss_0'].shape)
    print(len(logits), logits['logits_0'].shape)