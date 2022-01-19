from typing import Dict, List, Optional, Any, Union

from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertModel

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits, get_lengths_from_binary_sequence_mask, viterbi_decode
from allennlp.training.metrics.srl_eval_scorer import DEFAULT_SRL_EVAL_PATH, SrlEvalScorer
from allennlp.models.srl_util import convert_bio_tags_to_conll_format
from allennlp.training.metrics import CategoricalAccuracy

@Model.register("nombank-sense-srl-bert")
class NomSenseSRLBert(Model): # Model inherits from torch.nn.Module and Registrable
    """
    # Parameters
    
    vocab: `Vocabulary`, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    model: `Union[str, BertModel]`, required
        A string describing the BERT model to load or an already constructed BertModel.
    initializer: `InitializerApplicator`, optional (defeault=`InitializerApplicator()`)
        Used to initialize the model parameters.
    regularizer: `RegularizerApplicator`, optional (default=`None`)
        If provided, will be used to calculate the regularization penalty during training.
    label_smoothing: `float`, optional (default = 0.0)
        Whether or not to use label smoothing on labels when computing cross entropy loss.
    ignore_span_metric: `bool`, optional (default = False)
        Whether to calculate span loss, which is irrelevant when predicting BIO for Open Information Extraction.  
    srl_eval_path: `str`, optional (default=`DEFAULT_SRL_EVAL_PATH`)
        The path to the srl-eval.pl script. By default, will use the srl-eval.pl included with allennlp,
        which is located at allennlp-models/allennlp_models/syntax/srl/srl-eval.pl. If `None`, srl-eval.pl is not used. 

    """
    def __init__(
            self,
            vocab: Vocabulary,
            bert_model: Union[str, BertModel],
            embedding_dropout: float = 0.0,
            initializer: InitializerApplicator = InitializerApplicator(),
            regularizer: Optional[RegularizerApplicator] = None,
            label_smoothing: float = None,
            ignore_span_metric: bool = False,
            srl_eval_path: str = DEFAULT_SRL_EVAL_PATH,
            ) -> None:
        super(NomSenseSRLBert, self).__init__(vocab, regularizer)
        
        if isinstance(bert_model, str):
            self.bert_model = BertModel.from_pretrained(bert_model)
        else:
            self.bert_model = bert_model

        self.num_classes = self.vocab.get_vocab_size("labels") 
        self.sense_classes = self.vocab.get_vocab_size("sense_labels")

        if srl_eval_path is not None:
            # For span based evaluation, do not consider labels for predicate.
            # But in Nombank pre-processing, we did not label predicate in tags.
           self.span_metric = SrlEvalScorer(srl_eval_path, ignore_classes=[]) 
        else:
            self.span_metric = None
        
        self.tag_projection_layer = Linear(self.bert_model.config.hidden_size, self.num_classes)
        self.sense_projection_layer = Linear(self.bert_model.config.hidden_size, self.sense_classes)
        self.embedding_dropout = Dropout(p=embedding_dropout)
        self.sense_loss_fxn = torch.nn.CrossEntropyLoss()
        self.sense_accuracy = CategoricalAccuracy()
        self._label_smoothing = label_smoothing
        self.ignore_span_metric = ignore_span_metric
        initializer(self)

    def forward(
            self,
            tokens: Dict[str, torch.Tensor],
            nom_indicator: torch.Tensor,
            metadata: List[Any],
            tags: torch.LongTensor = None,
            sense: torch.LongTensor = None,
    ):
        """
        # Parameters

        tokens: Dict[str, torch.Tensor], required
            The output of `TextField.as_array()`, which should typically be passed directly to a 
            `TextFieldEmbedder`. For this model, this must be a `SingleIdTokenIndexer` which
            indexes wordpieces from the BERT vocabulary.
        nom_indicator: torch.LongTensor, required.
            An integer `SequenceFeatureField` representation of the position of the nominal
            in the sentence. Shape is (batch_size, num_tokens) and can be all zeros, if the
            sentence has no nominal predicate.
        tags: torch.LongTensor, optional (default = None)
            Torch tensor representing sequence of integer gold class labels of shape `(batch_size, num_tokens)`. 
        metadata: `List[Dict[str, Any]]`, optional (default = None)
            metadata containing the original words of the sentence, the nominal to compute
            the frame for, and start offsets to convert wordpieces back to a sequnce of words.
        sense : `torch.LongTensor`, optional (default = None)
            Torch tensor representing sense of the instance predicate. 
            Of shape `(batch_size, )`
        # Returns

        output dictionary consisting of:
        tag_logits: torch.FloatTensor
            A tensor of shape `(batch_size, num_tokens, tag_vocab_size)` representing
            unnormalized log probabilities of the tag classes.
        tag_class_probabilities: torch.FloatTensor
            A tensor of shape `(batch_size, num_tokens, tag_vocab_size)` representing
            a distribution of the tag classes per word
        sense_logits: torch.FloatTensor
            A tensor of shape `(batch_size, sense_vocab_size)` representing
            unnormalized log probabilities of the sense classes.
        sense_class_probabilities: torch.FloatTensor
            A tensor of shape `(batch_size, sense_vocab_size)` representing
            a distribution of the tag classes per word
        loss: torch.FloatTensor, optional
            A scalar loss to be optimized, during training phase.
        """
        mask = get_text_field_mask(tokens)
        bert_embeddings, _ = self.bert_model(
                input_ids=tokens["tokens"], #util.get_token_ids_from_text_field_tensors(tokens),
                token_type_ids=nom_indicator,
                attention_mask=mask,
                output_all_encoded_layers=False
                )
        embedded_text_input = self.embedding_dropout(bert_embeddings)
        batch_size, sequence_length, _ = embedded_text_input.size()
        tag_logits = self.tag_projection_layer(embedded_text_input)

        reshaped_tag_log_probs = tag_logits.view(-1, self.num_classes)
        tag_class_probabilities = F.softmax(reshaped_tag_log_probs, dim=-1).view(
                [batch_size, sequence_length, self.num_classes]
        )

        words, nominals, offsets, nom_indices = zip(*[(x["words"], x["nominal"], x["offsets"], x["nom_index"]) for x in metadata])
        
        sense_logits = self.sense_projection_layer(embedded_text_input)
        sense_logits_list = []
        for i, idx in enumerate(nom_indices):
            sense_logits_list.append(sense_logits[i][idx[0]])
        sense_logits = torch.stack(sense_logits_list)
        sense_class_probabilities = F.softmax(sense_logits, dim=-1).view([batch_size, self.sense_classes])

        output_dict = {"tag_logits": tag_logits, "sense_logits": sense_logits, "tag_class_probabilities": tag_class_probabilities, "sense_class_probabilities": sense_class_probabilities}
        # Retain the mask in the output dictionary so we can remove padding
        # when we do viterbi inference in self.make_output_human_readable.
        output_dict["mask"] = mask
        # Add in offsets to compute un-wordpieced tags. Hyphens will be separated.
        output_dict["words"] = list(words)
        output_dict["nominal"] = list(nominals)
        output_dict["wordpiece_offsets"] = list(offsets)
        output_dict["nominal_indices"] = nom_indices

        if tags is not None:
            sense_loss = 0
            if sense is not None:
                self.sense_accuracy(sense_logits, sense)
                sense_loss = self.sense_loss_fxn(sense_logits, sense.long().view(-1))
            tag_loss = sequence_cross_entropy_with_logits(
                    tag_logits, tags, mask, label_smoothing=self._label_smoothing
                    )
            if not self.ignore_span_metric and self.span_metric is not None and not self.training:
                batch_nom_indices = [
                        # Does not account for when nominal is entire hyphenated word. TODO
                        example_metadata["nom_index"][0] for example_metadata in metadata
                ]
                batch_sentences = [example_metadata["words"] for example_metadata in metadata]
                # Get BIO tags from make_output_human_readable()
                batch_bio_predicted_tags = self.decode(output_dict).pop("tags")
                batch_conll_predicted_tags = [
                        convert_bio_tags_to_conll_format(tags) for tags in batch_bio_predicted_tags
                ]
                batch_bio_gold_tags = [
                        example_metadata["gold_tags"] for example_metadata in metadata
                ]
                batch_conll_gold_tags = [
                        convert_bio_tags_to_conll_format(tags) for tags in batch_bio_gold_tags
                ]
               
                self.span_metric(
                        batch_nom_indices,
                        batch_sentences,
                        batch_conll_predicted_tags,
                        batch_conll_gold_tags,
                )
            output_dict["loss"] = sense_loss + tag_loss
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        """
        Performs constrained viterbi decoding on class probabilities output from the `forward` function. 
        Constraints are that the output tag must be a valid BIO sequence.

        Note that BIO sequence is decoded atop the wordpieces rather than the words.
        This yields higher performance also because the model is trained to perform 
        tagging on the wordpieces, not the words.
        """

        tag_predictions = output_dict["tag_class_probabilities"]
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()

        if tag_predictions.dim() == 3:
            tag_predictions_list = [
                    tag_predictions[i].detach().cpu() for i in range(tag_predictions.size(0))
            ]
        else:
            tag_predictions_list = [tag_predictions]

        wordpiece_tags = []
        word_tags = []
        transition_matrix, start_transitions = self.get_viterbi_pairwise_potentials_and_start_transitions()

        for predictions, length, offsets in zip(
                tag_predictions_list, sequence_lengths, output_dict["wordpiece_offsets"]
            ):
            max_likelihood_sequence, _ = viterbi_decode(
                    predictions[:length], transition_matrix, allowed_start_transitions=start_transitions
                    ) # why predictions[:length]? how do we know the last part is what we remove?
            tags = [
                    self.vocab.get_token_from_index(x, namespace="labels")
                    for x in max_likelihood_sequence
                    ]

            wordpiece_tags.append(tags)
            word_tags.append([tags[i] for i in offsets])
        output_dict["wordpiece_tags"] = wordpiece_tags
        output_dict["tags"] = word_tags
        
        sense_predictions = output_dict["sense_class_probabilities"]
        if sense_predictions.dim() == 2:
            sense_predictions_list = [sense_predictions[i] for i in range(sense_predictions.shape[0])]
        else:
            sense_predictions_list = [sense_predictions]

        sense_classes = []
        for sense_prediction in sense_predictions_list:
            label_idx = sense_prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary("sense_labels").get(label_idx, str(label_idx))
            sense_classes.append(label_str)
        output_dict["sense"] = sense_classes

        return output_dict

    def get_metrics(self, reset: bool = False):
        if self.ignore_span_metric:
            return {}
        else:
            metric_dict = self.span_metric.get_metric(reset=reset)
            sense_accuracy = self.sense_accuracy.get_metric(reset)
            return_dict = {x:y for x, y in metric_dict.items() if "overall" in x}
            return_dict["sense-accuracy"] = sense_accuracy
            return_dict["combined-score"] = return_dict["f1-measure-overall"] * sense_accuracy
            return return_dict

    def get_viterbi_pairwise_potentials_and_start_transitions(self):
        """
        Generate a matrix of pairwise transition potentials for the BIO labels.
        The only constraint implemented here is that I-XXX labels must be prededed
        by either an idential I-XXX tag or a B-XXX tag. In order to achieve this
        constraing, pairs of labels which do not satisfy this constraint have a
        pairwise potential of -inf.


        In the BIO sequence, we cannot start a sequence with any I-XXX tag.

        
        # Returns

        transition_matrix: torch.Tensor
            Of size (num_labels, num_labels): matrix of pairwise potentials.
        start_transitions: torch.Tensor
            The pairwise potentials between a START token and the first
            token of the sequence.
        """
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)
        transition_matrix = torch.zeros([num_labels, num_labels])
        start_transitions = torch.zeros(num_labels)

        for i, previous_label in all_labels.items():
            if previous_label[0] == "I":
                start_transitions[i] = float("-inf")
            for j, label in all_labels.items():
                # I-XXX labels can only be preceded by themselves or their corresp B-XXX tag.
                if i != j and label[0] == "I" and not previous_label=="B"+label[1:]: 
                    transition_matrix[i, j] = float("-inf")
        return transition_matrix, start_transitions

    default_predictor = "nombank-semantic-role-labeling"
