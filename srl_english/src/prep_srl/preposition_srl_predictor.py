from typing import List, Dict

import numpy
from overrides import overrides
from spacy.tokens import Doc

from allennlp.common.util import JsonDict, sanitize, group_by_count
from allennlp.predictors.predictor import Predictor
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.tokenizers.token import Token

@Predictor.register("preposition-semantic-role-labeling")
class PrepositionSemanticRoleLabelerPredictor(Predictor):
  '''
  Predictor for the preposition BERT-based SRL model.
  '''

  def __init__(
      self, model: Model, dataset_reader: DatasetReader, language: str="en_core_web_sm"
  ) -> None:
      super().__init__(model, dataset_reader)
      self._tokenizer = SpacyWordSplitter(language=language, pos_tags=True)

  def predict(self, sentence: str) -> JsonDict:
    '''
    Predicts the semantic roles of the supplied sentence, with respect to a preposition,
    and returns a dictionary with the results:
    ```
    {"words": [...],
     "prepositions": [
       {"prepositions": "...", "description": "...", "tags": [...]},
       ...
       {"prepositions": "...", "description": "...", "tags": [...]},
     ]}
    ```

    # Parameters

    sentence: `str`
      The sentence to parse via preposition srl.

    # Returns

    A dictionary representation of the preposition semantic roles of the sentence.
    '''
    return self.predict_json({"sentence": sentence})

  def predict_tokenized(self, tokenized_sentence: List[str]) -> JsonDict:
    '''
    # Parameters

    tokenized_sentence: `List[str]`
      The sentence tokens to parse.

    # Returns

    A dictionary representation of the nominal semantic roles of the sentence.
    '''

    spacy_doc = Doc(self._tokenizer.spacy.vocab, words=tokenized_sentence)
    for pipe in filter(None, self._tokenizer.spacy.pipeline):
      pipe[1](spacy_doc)

    tokens = [token for token in spacy_doc]
    instances = self.tokens_to_instances(tokens)

    if not instances:
        return sanitize({"prepositions": [], "words": tokens})

    return self.predict_instances(instances)

  @staticmethod
  def make_srl_string(words: List[str], tags: List[str]) -> str:
    frame = []
    chunk = []

    for (token, tag) in zip(words, tags):
      if tag.startswith("I-"):
        chunk = []

      if tag.startswith("B-"):
        chunk.append(tag[2:] + ": " + token)
      elif tag == "O":
        frame.append(token)

    if chunk:
      frame.append("[" + " ".join(chunk) + "]")

    return " ".join(frame)

  def _json_to_instance(self, json_dict: JsonDict):
    raise NotImplementedError("The SRL model uses a different API for creating instances.")

  def tokens_to_instances(self, tokens):
    '''
    # Parameters

    tokens: `List[str]`, required
      List of tokens of the original sentence.
    '''
    
    words = [token.text for token in tokens]
    instances: List[Instance] = []
    tokens_list = [(i, word) for (i, word) in enumerate(tokens)]
    skip_once = False
    skip_twice = False
    for (i, word) in tokens_list: # check for multiword prepositions
      if skip_once:
        skip_once = False
        continue
      if skip_twice:
        skip_twice = False
        skip_once = True
        continue
      # print(word, ", ", word.pos_)
      if word.pos_.strip() == "ADP": # encounter the start of the preposition
        if (i + 1) < len(tokens_list) and tokens_list[i + 1][1].pos_ == "ADP": 
          if (i + 2) < len(tokens_list) and tokens_list[i + 2][1].pos_ == "ADP": 
            prep_indices = [0 if j < i or j > (i + 2) else 1 for j in range(len(tokens_list))] # 3-word preposition
            skip_twice = True
          else: 
            prep_indices = [0 if j < i or j > (i + 1) else 1 for j in range(len(tokens_list))] # 2-word preposition
            skip_once = True
        else: 
          prep_indices = [0 if j != i else 1 for j in range(len(tokens_list))] # 1-word preposition
        # print('inside. prep_indices: ', prep_indices)
        instance = self._dataset_reader.text_to_instance(tokens, prep_indices, None, None) 
        instances.append(instance)
    return instances

  def _sentence_to_srl_instances(self, json_dict: JsonDict) -> List[Instance]:
    '''
    Need to run model forward for every detected preposition in the sentence, so for a single sentence
    generate a `List[Instance]` where the length of the instance list corresponds to the number of prepositions in the sentence.
    Expects input in the original format.

    # Parameters

    json_dict: `JsonDict`, required
      This JSON must look like `{"sentence": "..."}`.

    # Returns

    instances: `List[Instance]`
      One instance per preposition.
    '''

    sentence = json_dict["sentence"]
    tokens = self._tokenizer.split_words(sentence)
    return self.tokens_to_instances(tokens) 

  @overrides
  def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
    '''
    Perform JSON-to-JSON prediction.
    '''

    batch_size = len(inputs)
    instances_per_sentence = [self._sentence_to_srl_instances(json) for json in inputs]

    flattened_instances = [instance for sentence_instances in instances_per_sentence for instance in sentence_instance]

    if not flattened_instances:
      return sanitize([{"prepositions": [], "words": self._tokenizer.split_words(x["sentence"])} for x in inputs])

    # Batch instanced and check last batch for padded elements if number of instances is not a perfect multiple of the batch size
    batched_instanced = group_by_count(flattened_instances, batch_size, None)
    batched_instances[-1] = [instance for instance in batched_instances[-1] if instance is not None]

    outputs: List[Dict[str, numpy.ndarray]] = []
    for batch in batched_instances:
      outputs.extend(self._model.forward_on_instances(batch))

    preps_per_sentence = [len(sent) for sent in instances_per_sent]
    return_dicts: List[JsonDict] = [{"prepositions": []} for x in inputs]

    output_index = 0
    for sentence_index, prep_count in enumerate(preps_per_sentence):
      if prep_count == 0:
        # if sentence has no prepositions, just return the tokenization.
        original_text = self._tokenizer.split_words(inputs[sentence_index]["sentence"])

        return_dicts[sentence_index]["words"] = original_text
        continue

      for _ in range(prep_count):
        output = outputs[output_index]
        words = output["words"]
        tags = output["tags"]
        description = self.make_srl_string(words, tags) 
        return_dicts[sentence_index]["words"] = words
        return_dicts[sentence_index]["prepositions"].append({"preposition": output["preposition"], "predicate_index": output["preposition_indices"], "description": description, "tags":tags})
        output_index += 1

      return sanitize(return_dicts)

  def predict_instances(self, instances: List[Instance]) -> JsonDict:
    '''
    Perform prediction on instances of batch.
    '''
    outputs = self._model.forward_on_instances(instances)

    results = {"prepositions": [], "words": outputs[0]["words"]}
    for output in outputs:
      tags = output["tags"]
      description = self.make_srl_string(output["words"], tags)
      results["prepositions"].append({"preposition": output["preposition"], "predicate_index": output["preposition_indices"], "description": description, "tags": tags})
      
    return sanitize(results)

  @overrides
  def predict_json(self, inputs: JsonDict) -> JsonDict:
    '''
    Perform JSON-to-JSON prediction. Mainly just wraps work done by other functions.
    '''
    instances = self._sentence_to_srl_instances(inputs)

    if not instances:
      return sanitize({"prepositions": [], "words": self._tokenizer.split_words(inputs["sentence"])})

    return self.predict_instances(instances)
  

