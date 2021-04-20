'''
This provides a dataset reader for conll formatted preposition SRL data

include pos as metadata?
'''

import logging
from typing import Dict, List, Iterable, Tuple, Any

from overrides import overrides
from pytorch_pretrained_bert.tokenization import BertTokenizer

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token


logger = logging.getLogger(__name__)

def separate_hyphens(og_sentence: List[str]):
    new_sentence = []
    new_indices = []
    i = 0
    for word in og_sentence:
        broken_h_indices = []
        h_idx = word.find('-')
        bslash_idx = word.find('/')
        h_bs_idx = min(h_idx, bslash_idx) if h_idx>=0 and bslash_idx>=0 else max(h_idx, bslash_idx)
        prev_h_bs_idx = -1
        while h_bs_idx > 0:
            # subsection = word[prev_h_bs_idx+1:h_bs_idx+1]
            subsection = word[prev_h_bs_idx+1:h_bs_idx]
            broken_h_indices.append(i) # end of word before hyphen
            broken_h_indices.append(i+1) # end of hyphen
            new_sentence.append(subsection)
            new_sentence.append(word[h_bs_idx])
            prev_h_bs_idx = h_bs_idx
            h_idx = word.find('-', h_bs_idx+1)
            bslash_idx = word.find('/', h_bs_idx+1)
            h_bs_idx = min(h_idx, bslash_idx) if h_idx>=0 and bslash_idx>=0 else max(h_idx, bslash_idx)
            i += 2
        if not (prev_h_bs_idx == len(word)-1):
            subsection = word[prev_h_bs_idx+1:]
            new_sentence.append(subsection)
            broken_h_indices.append(i)
            i += 1
        new_indices.append(broken_h_indices)
    return new_sentence, new_indices


def _convert_tags_to_wordpiece_tags(new_tags: List[int], end_offsets: List[int]):
    """
    Converts a series of BIO tags to account for a wordpiece tokenizer,
    extending/modifying BIO tags where apropriate to deal with words which
    are split into multiple wordpieces by the tokenizer.

    # Parameters

    new_tags: `List[str]`
        The BIO formatted tags to convert to BIO tags for wordpieces. 
        Corresponds to hyphen-separated sentence, not original sentence.
    end_offsets: `List[int]`
        The wordpiece offsets.

    # Returns

    The new BIO tags.
    """
    wordpieced_tags = []
    j = 0
    for i, offset in enumerate(end_offsets):
        tag = new_tags[i]
        is_o = tag=="O"
        is_start = True
        while j < offset:
            if is_o:
                wordpieced_tags.append("O")
            elif tag.startswith("I"):
                wordpieced_tags.append(tag)
            elif is_start and tag.startswith("B"):
                wordpieced_tags.append(tag)
                is_start = False
            elif tag.startswith("B"):
                _, label = tag.split("-", 1)
                wordpieced_tags.append("I-" + label)
            j += 1
    return ["O"] + wordpieced_tags + ["O"]

def _convert_prep_indices_to_wordpiece_indices(prep_indices: List[int], end_offsets: List[int]):
    """
    Converts binary prep indicators to account for a wordpiece tokenizer.

    Parameters:
    
    prep_indices: `List[int]`
        The binary prep indicators, 0 for not the nom, 1 for the nom.
    end_offsets: `List[int]`
        The wordpiece end offsets, including for separated hyphenations.

    Returns:

    The new prep indices.
    """
    j = 0
    new_prep_indices = []
    for i, offset in enumerate(end_offsets): # For each word's offset (includes separated hyphenation)
        indicator = prep_indices[i] # 1 if word at i is prep, 0 if not.
        while j < offset:
            new_prep_indices.append(indicator) # Append indicator over length of wordpieces for word.
            j += 1

    # Add 0 indicators for cls and sep tokens.
    return [0] + new_prep_indices + [0]


@DatasetReader.register("preposition_srl")
class SrlReader(DatasetReader):
  '''
  This DatasetReader is designed to read in the Streusle data that has been converted to
  self-defined "span" format. This dataset reader specifically will read the data into a BIO format.
  It returns a dataset of instances with the following fields:

  tokens: `TextField`
    The tokens in the sequence.
  prep_indicator: `SequenceLabelField`
    A sequence of binary indicators for whether the word(s) is the preposition predicate for this frame.
  tags: `SequenceLabelField`
    A sequence of argument tags for the given preposition in a BIO format. 
  supersense1: `LabelField`
    A label for the first supersense expressed by the predicate.
  supersense2: `LabelField`
    A label for the second supersense expressed by the predicate. 

  # Parameters

  token_indexers: `Dict[str, TokenIndexer]`, optional
    We use this for both the premise and hypothesis.
    Default is `{"tokens": SingleIdTokenIndexer()}`.
  bert_model_name: `Optional[str]`, (default=None)
    The BERT model to be wrapped. If you specify a bert_model here, the BERT model will be used 
    throughout to expand tags and preposition indicator. If not, tokens will be indexed regularly
    with token_indexers.

  # Returns

  A `Dataset` or `Instances` for preposition Semantic Role Labeling.
  '''

  def __init__(self, token_indexers: Dict[str, TokenIndexer] = None, bert_model_name: str = None) -> None:
      super().__init__()
      self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
      if bert_model_name is not None:
          self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
          self.lowercase_input = "uncased" in bert_model_name 
      else: 
          self.bert_tokenizer = None
          self.lowercase_input = False

  def _wordpiece_tokenize_input(
            self, tokens: List[str]
            ) -> Tuple[List[str], List[int], List[int]]:
        """
        Convert a list of tokens to wordpiece tokens and offsets, as well as
        adding BERT CLS and SEP tokens to the beginning and end of the 
        sentence. The offsets will also point to sub-words inside hyphenated
        tokens. 
        For example:
        `stalemate` will be bert tokenized as ["stale", "##mate"].
        `quick-stalemate` will be bert tokenized as ["quick", "##-", "##sta", "##lem", "##ate"]
        We will want the tags to be at the finst granularity specified, like
        [B-GOV, I-GOV, B-OBJ, I-OBJ, I-OBJ]. The offsets will 
        correspond to the first word out of each hyphen chunk, even if the
        entire initial token is one argument. In this example, offsets would
        be [0, 2]
        # Returns
        wordpieces: List[str]
            The BERT wordpieces from the words in the sentence.
        end_offsets: List[int]
            Indices into wordpieces such that `[wordpieces[i] for i in end_offsets]` 
            results in the end wordpiece of each (separated) word chosen.
        start_offsets: List[int]
            Indices into wordpieces such that `[wordpieces[i] for i in start_offsets]`
            results in the start wordpiece of each word being chosen.
        """

        word_piece_tokens: List[str] = []
        end_offsets = []
        start_offsets = []
        cumulative = 0
        for token in tokens:
            if self.lowercase_input:
                token = token.lower()
            word_pieces = self.bert_tokenizer.wordpiece_tokenizer.tokenize(token)
            start_offsets.append(cumulative + 1) # +1 because we add the starting "[CLS]" token later.
            cumulative += len(word_pieces)
            end_offsets.append(cumulative)
            word_piece_tokens.extend(word_pieces)

        wordpieces = ["[CLS]"] + word_piece_tokens + ["[SEP]"]

        return wordpieces, end_offsets, start_offsets
  
  @overrides
  def _read(self, file_path: str):
    file_path = cached_path(file_path)
    logger.info("Reading SRL instances from dataset as %s", file_path)
    srl_data = self.read_prep_srl(file_path)
    for (sentence, predicate_location, supersense1, supersense2, tags, filename) in srl_data:
      tokens = [Token(t) for t in sentence]
      prep_indicator = [1 if "PREP" in tag else 0 for tag in tags]
      yield self.text_to_instance(tokens, prep_indicator, supersense1, supersense2, tags, filename)

  def read_prep_srl(self, filename):
    '''
    This process reads in the preposition SRL data in span format, and converts it to BIO format.

    example input line:
    reviews_086839_0002_0 One of the worst experiences I 've ever had with a auto repair shop . ||| p.QuantityItem p.Whole B-GOV B-PREP B-OBJ I-OBJ I-OBJ O O O O O O O O O O
    '''

    f = open(filename, "r")
    data = []
    for line in f.readlines():
      str_list = line.strip().split()
      separator_index = str_list.index("|||")
      filename = str_list[0]
      sentence = str_list[1:separator_index]
      supersense1 = str_list[separator_index + 1]
      supersense2 = str_list[separator_index + 2]
      tags = str_list[(separator_index + 3):]
      predicate_location = str_list.index("B-PREP")
      assert len(tags) == len(sentence)
      data.append((sentence, predicate_location, supersense1, supersense2, tags, filename))
    f.close()
    return data

  def text_to_instance(self, tokens: List[Token], prep_label: List[int], supersense1: str, supersense2: str, tags: List[str]=None, filename: str=None) -> Instance:
    '''
    We take the original sentence, `pre-tokenized` input as tokens here, as well as the preposition indices.
    The preposition label is a [one hot] binary vector, the same length as the tokens, indicating the position to find arguments for.
    The tags are BIO labels for the tokens.
    '''

    metadata_dict: Dict[str, Any] = {}
    new_sentence, new_indices = separate_hyphens([t.text for t in tokens])
    new_prep_label = [0 for _ in new_sentence]
    for idx, indicator in enumerate(prep_label):
        if indicator == 1:
            for new_idx in new_indices[idx]:
                new_prep_label[new_idx] = 1
    prep_label = new_prep_label

    if self.bert_tokenizer is not None:
      wordpieces, end_offsets, start_offsets = self._wordpiece_tokenize_input(
          [t.text for t in tokens]
      )
      # end_offsets and start_offsets are computed to correspond to sentence
      new_prep = _convert_prep_indices_to_wordpiece_indices(prep_label, end_offsets)
      metadata_dict["offsets"] = start_offsets
      text_field = TextField(
          [Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in wordpieces],
          token_indexers=self._token_indexers,
      )
      prep_indicator = SequenceLabelField(new_prep, text_field)
    else: # without a bert tokenizer, just give it the tokens and corresponding information
      text_field = TextField(tokens, seld._token_indexers)
      prep_indicator = SequenceLabelField(prep_label, text_field)

    fields: Dict[str, Field] = {}
    fields["tokens"] = text_field
    fields["prep_indicator"] = prep_indicator

    if all(x == 0 for x in prep_label):
      prep = None
      prep_index = None
    else: 
      prep_index = [i for i in range(len(prep_label)) if prep_label[i] == 1]
      prep = ""
      for p_idx in prep_index: # prep_index is indexed to words
        prep += tokens[p_idx].text
      
    metadata_dict["words"] = [x.text for x in tokens]
    metadata_dict["preposition"] = prep
    metadata_dict["prep_index"] = prep_index
    metadata_dict["filename"] = filename

    if tags:
      new_tags = ['O' for _ in new_sentence]
      for idx, old_tag in enumerate(tags):
          for new_idx in new_indices[idx]:
              new_tags[new_idx] = old_tag
      tags = new_tags
      if self.bert_tokenizer is not None:
        wordpieced_tags = _convert_tags_to_wordpiece_tags(tags, end_offsets)
        fields["tags"] = SequenceLabelField(wordpieced_tags, text_field)
      else:
        fields["tags"] = SequenceLabelField(tags, text_field)
      metadata_dict["gold_tags"] = tags

    # this is left here since we might eventually want to add the functionality of incorporating senses into the prediction process
    #supersense1_label = LabelField()
    #supersense2_label = LabelField()

    fields["metadata"] = MetadataField(metadata_dict)
    return Instance(fields)

  
