import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel


class BertClassifier(BertPreTrainedModel):
    def __init__(self, config, num_labels, vocab) -> None:
        super(BertClassifier, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.vocab = vocab
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, output_all_encoded_layers=False):
        # pooled_output is not same hidden vector corresponds to first token from last encoded layers
        attention_mask = input_ids.ne(self.vocab.to_indices(self.vocab.padding_token)).float()
        if output_all_encoded_layers:
            encoded_layers, pooled_output = self.bert(input_ids, None, attention_mask,
                                                      output_all_encoded_layers=output_all_encoded_layers)
            encoded_layers = [encoded_layer[:, 0, :] for encoded_layer in encoded_layers]
            encoded_layers.append(pooled_output)
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            return logits, encoded_layers
        else:
            _, pooled_output = self.bert(input_ids, None, attention_mask,
                                         output_all_encoded_layers=output_all_encoded_layers)
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            return logits, pooled_output


class Detector(nn.Module):
    def __init__(self, input_features_dim, num_classes):
        super(Detector, self).__init__()
        self._ops = nn.Sequential(nn.Linear(input_features_dim, input_features_dim // 2),
                                  nn.ReLU(),
                                  nn.Dropout(),
                                  nn.Linear(input_features_dim // 2, num_classes),
                                  nn.Dropout())

    def forward(self, x):
        score = self._ops(x)
        return score
