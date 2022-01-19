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
            subsection = word[prev_h_bs_idx+1:h_bs_idx+1]
            broken_h_indices.append(i)
            new_sentence.append(subsection)
            prev_h_bs_idx = h_bs_idx
            h_idx = word.find('-', h_bs_idx+1)
            bslash_idx = word.find('/', h_bs_idx+1)
            h_bs_idx = min(h_idx, bslash_idx) if h_idx>=0 and bslash_idx>=0 else max(h_idx, bslash_idx)
            i += 1
        subsection = word[prev_h_bs_idx+1:]
        new_sentence.append(subsection)
        broken_h_indices.append(i)
        i += 1
        new_indices.append(broken_h_indices)
    return new_sentence, new_indices

def _convert_nom_indices_to_wordpiece_indices(nom_indices: List[int], end_offsets: List[int]):
    """
    Converts binary nom indicators to account for a wordpiece tokenizer.

    Parameters:

    nom_indices: `List[int]`
        The binary nom indicators: 0 for not the nom, 1 for the nom.
    end_offsets: `List[int]`
        The wordpiece end offsets, including for separated hyphenations.

    Returns:

    The new nom indices.
    """
    j = 0
    new_nom_indices = []
    for i, offset in enumerate(end_offsets): # For each word's offset (includes separated hyphenation)
        indicator = nom_indices[i] # 1 if word at i is part of nominal predicate, 0 if not.
        while j < offset: # Append indicator over lenth of wordpieces for word.
            new_nom_indices.append(indicator) 
            j += 1

    # Add 0 incidators for cls and sep tokens.
    return [0] + new_nom_indices + [0]

@DatasetReader.register("nombank-id")
class NombankReader(DatasetReader):
    """
    This DatasetReader is designed to read in the Nombank data that has been 
    converted to self-defined "span" format. This dataset reader specifically
    will take in the sentence, sense, and arguments. It will also break apart
    the sentence into separated hyphenations. It returns a dataset
    of instances with the following field:

    tokens: `TextField`
        The tokens in the sentence.
    pred_indicator: `SequenceLabelField`, optional
        A sequence of binary indicators for whether the word is part of a nominal
        predicate.
    metadata: `Dict[str, Any]`

    # Parameters

    token_indexer: `Dict[str, TokenIndexer]`, optional
        We use this for both the premise and the hypothesis.
        Default is `{"tokens": SingleIdTokenIndexer()}`
    bert_model_name: `Optional[str]`, (default=None)
        The BERT model to be wrapped. If you specify a bert_model here, the
        BERT model will be used throughout to expand nom indicator.
        If not, tokens will be indexed regularly with token_indexers

    # Returns

    A `Dataset` of `Instances` for nominal predicate identification.
    """

    def __init__(
            self,
            token_indexers: Dict[str, TokenIndexer] = None,
            lazy:bool = False,
            bert_model_name: str = None,
            #**kwargs,
        ) -> None:
        super().__init__(lazy)
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
        Converts a list of tokens to wordpiece tokens and offsets, as well as
        adding BERT CLS and SEP tokens to the beginning and end of the sentence.
        The offsets will also point to sub-words inside hyphenated tokens.

        # Returns

        wordpieces: `List[str]`
            The BERT wordpieces from the words in the sentence.
        end_offsets: `List[int]`
            Indices into wordpieces such that `[wordpieces[i] for i in end_offsets]`
            results in the end wordpiece of each (separated) word chosen.
        start_offsets: `List[int]`
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
            start_offsets.append(cumulative+1) # +1 because we add the starting "[CLS]" token
            cumulative += len(word_pieces)
            end_offsets.append(cumulative)
            word_piece_tokens.extend(word_pieces)

        wordpieces = ["[CLS]"] + word_piece_tokens + ["[SEP]"]

        return wordpieces, end_offsets, start_offsets

    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)
        logger.info("Reading Nombank instances from dataset file at %s", file_path)
        predid_data = self.read_nombank_data(file_path)
        for (og_sentence, og_pred_locs, new_indices, new_sentence) in predid_data:
            og_tokens = [Token(t) for t in og_sentence]
            new_tokens = [Token(t) for t in new_sentence]
            preds_indicator = [0 for _ in new_sentence]
            for pred_loc in og_pred_locs:
                new_pred_loc = new_indices[pred_loc[0]]
                if len(pred_loc) > 1:
                    if pred_loc[1] >= len(new_pred_loc):
                        print('Faulty data point. Trying to access hyphenation that does not exist.')
                        continue
                    new_pred_loc = [new_pred_loc[pred_loc[1]]]
                
                preds_indicator[new_pred_loc[0]] = 1
                # If predicate is entire hyphenation, this line covers it.
            #print('NEW INSTANCE: TOKENS: ', new_tokens, '; preds_indicator: ', preds_indicator)
            yield self.text_to_instance(og_tokens, new_tokens, preds_indicator)

    def read_nombank_data(self, filename):
        """
        This process reads in the preprocessed Nombank data for nominal id.

        example input lines: 
        wsj/01/wsj_0199.mrg 2 Trinity said it plans to begin delivery in the first quarter of next year . ||| 01 10:10-rel 11:13-ARG1
        wsj/01/wsj_0199.mrg 2 Trinity said it plans to begin delivery in the first quarter of next year . ||| 01 6:6-rel 2:2*4:4-ARG0 5:5-Support
        
        its output:
        (
            og_sentence = ['Trinity', 'said', 'it', 'plans', ... 'year', '.'],
            og_nom_locs = [10, 6],
            new_indices = [0, 1, 2, 3...],
            new_sentence = ['Trinity', 'said', ... '.'],
        )
        """

        fin = open(filename, 'r')
        data = []
        last_file = ""
        last_sentnum = -1
        last_sentence = ""
        new_indices = []
        new_sentence = ""
        sentence_predicates = []
        for line in fin.readlines():
            str_list = line.strip().split()
            separator_index = str_list.index("|||")
            fileid = str_list[0]
            sentnum = str_list[1]
            # Get index of predicate. Predicate is always first argument.
            predicate_loc = str_list[separator_index+2][:str_list[separator_index+2].find(':')]
            sub_idx = predicate_loc.find('_')
            if sub_idx < 0:
                og_nom_loc = [int(predicate_loc)]
            else:
                og_nom_loc = [int(predicate_loc[:sub_idx]), int(predicate_loc[sub_idx+1:])]
            if fileid == last_file:
                if sentnum == last_sentnum:
                    sentence_predicates.append(og_nom_loc)
                    continue
                else:
                    data.append((last_sentence, sentence_predicates, new_indices, new_sentence))
                    last_sentnum = sentnum
                    last_sentence = str_list[2:separator_index]
                    new_sentence, new_indices = separate_hyphens(last_sentence)
                    sentence_predicates = [og_nom_loc]
            else:
                if len(sentence_predicates) > 0:
                    data.append((last_sentence, sentence_predicates, new_indices, new_sentence))
                last_file = fileid
                last_sentnum = sentnum
                last_sentence = str_list[2:separator_index]
                new_sentence, new_indices = separate_hyphens(last_sentence)
                sentence_predicates = [og_nom_loc]
        if len(sentence_predicates) > 0:
            data.append((last_sentence, sentence_predicates, new_indices, new_sentence))
        fin.close()
        return data
    
    def text_to_instance(
            self, og_tokens: List[Token], new_tokens: List[Token], pred_label: List[int]=None
        ) -> Instance:
        """
        We take original sentence, `pre-tokenized` input as tokens here, as well as the
        tokens corresponding to once tokenized and de-hyphenated. The predicate label is 
        a binary vector, the same length as new_tokens, indicating position(s) of the
        predicates of the sentence.
        """
        metadata_dict: Dict[str, Any] = {}
        fields: Dict[str, Field] = {}
        if self.bert_tokenizer is not None:
            wordpieces, end_offsets, start_offsets = self._wordpiece_tokenize_input(
                    [t.text for t in new_tokens]
            )
            # end_offsets and start_offsets are computed to correspond to sentence with separated hyphens.
            text_field = TextField(
                [Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in wordpieces],
                token_indexers=self._token_indexers,
            )
            metadata_dict["offsets"] = start_offsets
        else: # Without a bert tokenizer, just give it new tokens and corresponding info.
            text_field = TextField(new_tokens, token_indexers=self._token_indexers)

        fields["tokens"] = text_field
        metadata_dict["words"] = [x.text for x in new_tokens]

        if pred_label is not None:
            if self.bert_tokenizer is not None:
                pred_label = _convert_nom_indices_to_wordpiece_indices(pred_label, end_offsets)
            pred_indicator = SequenceLabelField(pred_label, text_field)
            fields["predicate_indicator"] = pred_indicator

        fields["metadata"] = MetadataField(metadata_dict)
        return Instance(fields)


