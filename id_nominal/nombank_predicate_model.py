from typing import Dict, List, Optional, Any, Union

from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertModel

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits, get_lengths_from_binary_sequence_mask#, viterbi_decode
from allennlp.training.metrics import F1Measure
from allennlp.models.srl_util import convert_bio_tags_to_conll_format

@Model.register("nombank-id-bert")
class NombankIdBert(Model):
    """
    # Parameters
    
    vocab: `Vocabulary`, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    model: `Union[str, BertModel]`, required
        A string describing the BERT model to load or an already constructed BertModel.
    initializer: `InitializerApplicator`, optional (defeault=`InitializerApplicator()`)
        Used to initialize the model parameters.

    """
    def __init__(
            self,
            vocab: Vocabulary,
            bert_model: Union[str, BertModel],
            embedding_dropout: float = 0.0,
            initializer: InitializerApplicator = InitializerApplicator(),
            regularizer: Optional[RegularizerApplicator] = None,
            #**kwargs
            ) -> None:
        super(NombankIdBert, self).__init__(vocab, regularizer)
        
        if isinstance(bert_model, str):
            self.bert_model = BertModel.from_pretrained(bert_model)
        else:
            self.bert_model = bert_model

        self.num_classes = 2 # Binary indicator: either 0 or 1
        self.span_metric = F1Measure(positive_label=1) 
        
        self.tag_projection_layer = Linear(self.bert_model.config.hidden_size, self.num_classes)

        self.embedding_dropout = Dropout(p=embedding_dropout)
        initializer(self)

    def forward(
            self,
            tokens: Dict[str, torch.Tensor],
            metadata: List[Any],
            predicate_indicator: torch.Tensor = None,
    ):
        """
        # Parameters

        tokens: Dict[str, torch.Tensor], required
            The output of `TextField.as_array()`, which should typically be passed directly to a 
            `TextFieldEmbedder`. For this model, this must be a `SingleIdTokenIndexer` which
            indexes wordpieces from the BERT vocabulary.
        predicate_indicator: torch.LongTensor, optional (default = None).
            An integer `SequenceFeatureField` representation of the position of the predicate
            in the sentence. Shape is (batch_size, num_tokens) and can be all zeros, if the
            sentence has no predicate.
        metadata: `List[Dict[str, Any]]`, optional (default = None)
            metadata containing the original words of the sentence, the location of predicate,
            and start offsets to convert wordpieces back to a sequnce of words.

        # Returns

        output dictionary consisting of:
        logits: torch.FloatTensor
            A tensor of shape `(batch_size, num_tokens)` representing
            unnormalized log probabilities of the token being a nominal predicate
        class_probabilities: torch.FloatTensor
            A tensor of shape `(batch_size, num_tokens, 2)` representing
            a distribution of the tag classes per word
        loss: torch.FloatTensor, optional
            A scalar loss to be optimized, during training phase.
        """
        mask = get_text_field_mask(tokens)
        bert_embeddings, _ = self.bert_model(
                input_ids=tokens["tokens"],
                token_type_ids=None,
                attention_mask=mask,
                output_all_encoded_layers=False
        )
        embedded_text_input = self.embedding_dropout(bert_embeddings)
        batch_size, sequence_length, _ = embedded_text_input.size()
        logits = self.tag_projection_layer(embedded_text_input)

        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view(
                [batch_size, sequence_length, self.num_classes]
        )

        output_dict = {"logits": logits, "class_probabilities": class_probabilities}
        # Retain the mask in the output dictionary so we can remove padding
        # when we do viterbi inference in self.make_output_human_readable.
        output_dict["mask"] = mask
        # Add in offsets to compute un-wordpieced tags. Hyphens will be separated.
        words, offsets = zip(*[(x["words"], x["offsets"]) for x in metadata])
        output_dict["words"] = list(words)
        output_dict["wordpiece_offsets"] = list(offsets)

        if predicate_indicator is not None:
            # print('PREDICATE_INDICATOR: ', predicate_indicator)
            loss = sequence_cross_entropy_with_logits(
                    logits, predicate_indicator, mask
            )
            if True: #self.training: # or not self.training?
                predicate_predictions = self.decode(output_dict).pop("wordpiece_indicator")
                # print('PREDICATE_PREDICTIONS: ', predicate_predictions)
                
                # max_seq_length = predicate_indicator.size()[-1]
                padded_predicate_predictions = []
                n = 0
                for prediction in predicate_predictions:
                    prediction_tensor = torch.stack(prediction)
                    padding_tensor = torch.zeros(sequence_length-len(prediction), 2)
                    padded_prediction = torch.cat([prediction_tensor, padding_tensor])
                    padded_predicate_predictions.append(padded_prediction)
                    n += 1
                padded_predicate_predictions = torch.stack(padded_predicate_predictions)

                # print('predicate_indicator shape: ', predicate_indicator.size())
                # print('padded_predicate_predictions shape: ', padded_predicate_predictions.size())
                # print('PADDED_PREDICATE_PREDICTIONS: ', padded_predicate_predictions)
                self.span_metric(
                        padded_predicate_predictions,
                        predicate_indicator,
                        mask
                )
            output_dict["loss"] = loss
        return output_dict

    @overrides
    def decode(
            self, output_dict: Dict[str, torch.Tensor]
            ) -> Dict[str, torch.Tensor]:

        all_predictions = output_dict["class_probabilities"]
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()

        if all_predictions.dim() == 3:
            predictions_list = [
                    all_predictions[i].detach().cpu() for i in range(all_predictions.size(0))
                    ]
        else:
            predictions_list = [all_predictions]

        wordpiece_indicator = []
        word_indicator = []
        
        for predictions, length, offsets in zip(
                predictions_list, sequence_lengths, output_dict["wordpiece_offsets"]
                ):
            prediction_truncated = predictions[:length]
            # predicate_indicator = [torch.argmax(class_probs) for class_probs in prediction_truncated]
            wordpiece_indicator.append([x for x in prediction_truncated])
            word_indicator.append([torch.argmax(prediction_truncated[i]) for i in offsets])
        output_dict["wordpiece_indicator"] = wordpiece_indicator
        output_dict["predicate_indicator"] = word_indicator
        return output_dict

    def get_metrics(self, reset: bool = False):
        metric_dict = self.span_metric.get_metric(reset=reset)
        return {"precision":metric_dict[0], "recall":metric_dict[1], "fscore": metric_dict[2]}

    
    default_predictor = "nombank-id"
