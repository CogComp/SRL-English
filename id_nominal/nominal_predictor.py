from typing import List, Dict

import numpy
from overrides import overrides
from spacy.tokens import Doc

from allennlp.common.util import JsonDict, sanitize, group_by_count
from allennlp.predictors.predictor import Predictor
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

from id_nominal.nombank_reader import separate_hyphens

@Predictor.register("nombank-id")
class NominalIdPredictor(Predictor):
    """
    Predictor for the nominal BERT-based ID model.
    """

    def __init__(
            self, model: Model, dataset_reader: DatasetReader, language: str="en_core_web_sm"
            ) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language=language, pos_tags=True)

    def predict(self, sentence: str) -> JsonDict:
        """
        Predicts the semantic roles of the supplied sentence, with respect to a nominal,
        and returns a dictionary with the results:
        ```
        {"words": [...],
         "nominals": [...]}
        ```

        # Parameters

        sentence: `str`
            The sentence to identify nominal predicates from.

        # Returns

        A dictionary representation of the nominal identifications of the sentence.
        """
        # TODO replace all parentheses and other brackets? maybe tokenizer already does it?
        return self.predict_json({"sentence": sentence})

    def predict_tokenized(self, tokenized_sentence: List[str]) -> JsonDict:
        """
        # Parameters

        tokenized_sentence: `List[str]`
            The sentence tokens to parse.

        # Returns

        A dictionary representation of the nominal semantic roles of the sentence.
        """
        spacy_doc = Doc(self._tokenizer.spacy.vocab, words=tokenized_sentence)
        for pipe in filter(None, self._tokenizer.spacy.pipeline):
            pipe[1](spacy_doc) 

        tokens = [token for token in spacy_doc]
        instances = self.tokens_to_instances(tokens)

        if not instances:
            return sanitize({"nominals": [], "words": tokens})
        
        return self.predict_instances(instances)
    
    @overrides
    def _json_to_instance(self, json_dict: JsonDict):
        raise NotImplementedError("The SRL mdel uses a different API for creating instances.")

    def tokens_to_instances(self, tokens):
        """
        # Parameters

        tokens: `List[Token]`, required
            List of tokens of the original sentence, before hyphenated separation.
        """
        words = [token.text for token in tokens]
        new_sentence, new_indices = separate_hyphens(words)
        new_tokens = [Token(t) for t in new_sentence]
        instances: List[Instance] = []
        for i, word in enumerate(tokens):
            instance = self._dataset_reader.text_to_instance(tokens, new_tokens)
            instances.append(instance)
        return instances

    def _sentence_to_srl_instances(self, json_dict: JsonDict) -> List[Instance]:
        """
        Need to run model forward for every detected nominal in the sentence, so for
        a single sentence, generate a `List[Instance]` where the length of the ilist 
        corresponds to the number of nominals in the sentence. Expects input in the
        original format, and dehyphenates it to return instances.

        # Parameters

        json_dict: `JsonDict`, required
            This JSON must look like `{"sentence": "... "}`.

        # Returns

        instances: `List[Instance]`
            One instance per nominal.
        """
        sentence = json_dict["sentence"]
        tokens = self._tokenizer.split_words(sentence)
        return self.tokens_to_instances(tokens)

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        """
        Perform JSON-to-JSON predition.
        """
        batch_size = len(inputs)
        instances_per_sentence = [self._sentence_to_srl_instances(json) for json in inputs]

        flattened_instances = [
                instance
                for sentence_instances in instances_per_sentence
                for instance in sentence_instances
        ]

        if not flattened_instances:
            return sanitize(
                    [{"nominals": [], "words": self._tokenizer.split_words(x["sentence"])} for x in inputs]
                    )

        # Batch instances and check last batch for padded elements if number of instances
        # is not a perfect multiple of the batch size.
        batched_instances = group_by_count(flattened_instances, batch_size, None)
        batched_instances[-1] = [
                instance for instance in batched_instances[-1] if instance is not None
                ]
        outputs: List[Dict[str, numpy.ndarray]] = []
        for batch in batched_instances:
            outputs.extend(self._model.forward_on_instances(batch))
        
        return_dicts: List[JsonDict] = [{"words": output["words"] ,"nominals": output["predicate_indicator"]} for output in outputs]

        return sanitize(return_dicts)

    def predict_instances(self, instances: List[Instance]) -> JsonDict:
        """ 
        Perform prediction on instances of batch.
        """
        outputs = self._model.forward_on_instances(instances)

        results = {"nominals": outputs[0]["predicate_indicator"], "words": outputs[0]["words"]}
        return sanitize(results)

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        """
        Perform JSON-to-JSON prediction. Mainly just wraps work done by other functions.
        """
        instances = self._sentence_to_srl_instances(inputs)

        if not instances:
            return sanitize({"nominals": [], "words": self._tokenizer.split_words(inputs["sentence"])})

        return self.predict_instances(instances)
