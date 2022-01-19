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

from nominal_srl.nominal_srl_reader import separate_hyphens

@Predictor.register("nombank-sense-srl")
class NomSenseSRLPredictor(Predictor):
    """
    Predictor for the nominal BERT-based SRL model.
    """

    def __init__(
            self, model: Model, dataset_reader: DatasetReader, language: str="en_core_web_sm"
            ) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language=language, pos_tags=True)

    def predict(self, sentence: str, indices: List[int]) -> JsonDict:
        """
        Predicts the semantic roles of the supplied sentence, with respect to a nominal,
        and returns a dictionary with the results:
        ```
        {"words": [...],
         "nominals": [
            {"nominals": "...", "sense": "..", "description": "...", "tags": [...]},
            ...
            {"nominals": "...", "sense": "..", "description": "...", "tags": [...]},
         ]}
        ```

        # Parameters

        sentence: `str`
            The sentence to parse via nominal srl.

        # Returns

        A dictionary representation of the nominal semantic roles of the sentence.
        """
        # TODO replace all parentheses and other brackets? maybe tokenizer already does it?
        return self.predict_json({"sentence": sentence, "indices": indices})

    def predict_tokenized(self, tokenized_sentence: List[str], indices: List[int]) -> JsonDict:
        """
        # Parameters

        tokenized_sentence: `List[str]`
            The sentence tokens to parse.
        indices: `List[int]`
            The indices of the predicates to predict on.

        # Returns

        A dictionary representation of the nominal semantic roles of the sentence.
        """
        spacy_doc = Doc(self._tokenizer.spacy.vocab, words=tokenized_sentence)
        for pipe in filter(None, self._tokenizer.spacy.pipeline):
            pipe[1](spacy_doc) 

        tokens = [token for token in spacy_doc]
        instances = self.tokens_to_instances(tokens, indices)

        if not instances:
            return sanitize({"nominals": [], "words": tokens})
        
        return self.predict_instances(instances)

    @staticmethod
    def make_srl_string(words: List[str], tags: List[str]) -> str:
        frame = []
        chunk = []

        for (token, tag) in zip(words, tags):
            if tag.startswith("I-"):
                chunk.append(token)
            else:
                if chunk:
                    frame.append("[" + " ".join(chunk) + "]")
                    chunk = []

                if tag.startswith("B-"):
                    chunk.append(tag[2:] + ": " + token) 
                elif tag == "O":
                    frame.append(token)

        if chunk:
            frame.append("[" + " ".join(chunk) + "]")

        return " ".join(frame)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict):
        raise NotImplementedError("The SRL mdel uses a different API for creating instances.")

    def tokens_to_instances(self, tokens, indices):
        """
        # Parameters

        tokens: `List[Token]`, required
            List of tokens of the original sentence, before hyphenated separation.
        indices: `List[int]`, required
            List of indices corresponding to the predicates to predict on.
        """
        words = [token.text for token in tokens]
        new_sentence, new_indices = separate_hyphens(words)
        new_tokens = [Token(t) for t in new_sentence]
        instances: List[Instance] = []
        for index in indices:
            new_nom_idx = new_indices[index]
            nom_labels = [0 for _ in new_tokens]
            for new_i in new_nom_idx:
                nom_labels[new_i] = 1
            instance = self._dataset_reader.text_to_instance(tokens, new_tokens, nom_labels)
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
            This JSON must look like `{"sentence": "... ", "indices": [...]}`.

        # Returns

        instances: `List[Instance]`
            One instance per nominal.
        """
        sentence = json_dict["sentence"]
        indices = json_dict["indices"]
        tokens = self._tokenizer.split_words(sentence)
        return self.tokens_to_instances(tokens, indices)

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

        noms_per_sentence = [len(sent) for sent in instances_per_sentence]
        return_dicts: List[JsonDict] = [{"nominals": []} for x in inputs]

        output_index = 0
        for sentence_index, nom_count in enumerate(noms_per_sentence):
            if nom_count == 0:
                # If sentence has no nominals, just return the tokenization.
                original_text = self._tokenizer.split_words(inputs[sentence_index]["sentence"])

                return_dicts[sentence_index]["words"] = original_text
                continue

            for _ in range(nom_count):
                output = outputs[output_index]
                words = output["words"]
                tags = output["tags"]
                description = self.make_srl_string(words, tags)
                return_dicts[sentence_index]["words"] = words
                return_dicts[sentence_index]["nominals"].append(
                        {"nominal": output["nominal"], "sense": output["sense"], "predicate_index": output["nominal_indices"], "description": description, "tags": tags}
                        )
                output_index += 1
            
        return sanitize(return_dicts)

    def predict_instances(self, instances: List[Instance]) -> JsonDict:
        """ 
        Perform prediction on instances of batch.
        """
        outputs = self._model.forward_on_instances(instances)

        results = {"nominals": [], "words": outputs[0]["words"]}
        for output in outputs:
            tags = output["tags"]
            description = self.make_srl_string(output["words"], tags)
            results["nominals"].append(
                    {"nominal": output["nominal"], "sense": output["sense"], "predicate_index": output["nominal_indices"], "description": description, "tags": tags}
                    )
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
