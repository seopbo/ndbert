import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel


class BertClassifier(BertPreTrainedModel):
    def __init__(self, config, num_classes, vocab) -> None:
        super(BertClassifier, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        self.vocab = vocab
        self.init_weights()

    def forward(self, input_ids):
        # pooled_output is not same hidden vector corresponds to first token from last encoded layers
        attention_mask = input_ids.ne(
            self.vocab.to_indices(self.vocab.padding_token)
        ).float()
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs[1])
        logits = self.classifier(pooled_output)

        if self.config.output_hidden_states:
            all_hidden_states = [
                transformer_layer[:, 0, :] for transformer_layer in outputs[2][1:]
            ]
            all_hidden_states.append(pooled_output)
            return logits, all_hidden_states
        else:
            return logits, outputs[0][:, 0, :]
