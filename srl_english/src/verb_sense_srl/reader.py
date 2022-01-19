import logging
from typing import Dict, List, Iterable, Tuple, Any

from overrides import overrides
from pytorch_pretrained_bert.tokenization import BertTokenizer

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, OntonotesSentence
from bolt_srl.bolt import Bolt, BoltSentence

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

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
            broken_h_indices.append(i+1) # end of hyphe
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


def _convert_tags_to_wordpiece_tags(tags: List[str], offsets: List[int]) -> List[str]:
    """
    Converts a series of BIO tags to account for a wordpiece tokenizer,
    extending/modifying BIO tags where appropriate to deal with words which
    are split into multiple wordpieces by the tokenizer.

    This is only used if you pass a `bert_model_name` to the dataset reader below.

    Parameters
    ----------
    tags : `List[str]`
        The BIO formatted tags to convert to BIO tags for wordpieces
    offsets : `List[int]`
        The wordpiece offsets.

    Returns
    -------
    The new BIO tags.
    """
    new_tags = []
    j = 0
    for i, offset in enumerate(offsets):
        tag = tags[i]
        is_o = tag == "O"
        is_start = True
        while j < offset:
            if is_o:
                new_tags.append("O")

            elif tag.startswith("I"):
                new_tags.append(tag)

            elif is_start and tag.startswith("B"):
                new_tags.append(tag)
                is_start = False

            elif tag.startswith("B"):
                _, label = tag.split("-", 1)
                new_tags.append("I-" + label)
            j += 1

    # Add O tags for cls and sep tokens.
    return ["O"] + new_tags + ["O"]


def _convert_verb_indices_to_wordpiece_indices(verb_indices: List[int], offsets: List[int]): # pylint: disable=invalid-name
    """
    Converts binary verb indicators to account for a wordpiece tokenizer,
    extending/modifying BIO tags where appropriate to deal with words which
    are split into multiple wordpieces by the tokenizer.

    This is only used if you pass a `bert_model_name` to the dataset reader below.

    Parameters
    ----------
    verb_indices : `List[int]`
        The binary verb indicators, 0 for not a verb, 1 for verb.
    offsets : `List[int]`
        The wordpiece offsets.

    Returns
    -------
    The new verb indices.
    """
    j = 0
    new_verb_indices = []
    for i, offset in enumerate(offsets):
        indicator = verb_indices[i]
        while j < offset:
            new_verb_indices.append(indicator)
            j += 1

    # Add 0 indicators for cls and sep tokens.
    return [0] + new_verb_indices + [0]


@DatasetReader.register("sense-srl")
class SenseSRLReader(DatasetReader):
    """
    This DatasetReader is designed to read in the English OntoNotes v5.0 data
    for sense disambiguation and semantic role labelling. 
    It returns a dataset of instances with the following fields:

    tokens : ``TextField``
        The tokens in the sentence.
    verb_indicator : ``SequenceLabelField``
        A sequence of binary indicators for whether the word is the verb for this frame.
    sense: ``LabelField``
        Label of sense for verbal predicate instance.
    tags : ``SequenceLabelField``
        A sequence of Propbank tags for the given verb in a BIO format.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    domain_identifier: ``str``, (default = None)
        A string denoting a sub-domain of the Ontonotes 5.0 dataset to use. If present, only
        conll files under paths containing this domain identifier will be processed.
    bert_model_name : ``Optional[str]``, (default = None)
        The BERT model to be wrapped. If you specify a bert_model here, then we will
        assume you want to use BERT throughout; we will use the bert tokenizer,
        and will expand your tags and verb indicators accordingly. If not,
        the tokens will be indexed as normal with the token_indexers.

    Returns
    -------
    A ``Dataset`` of ``Instances`` for Semantic Role Labelling.
    """
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 domain_identifier: str = None,
                 lazy: bool = False,
                 bert_model_name: str = None) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._domain_identifier = domain_identifier

        if bert_model_name is not None:
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            self.lowercase_input = "uncased" in bert_model_name
        else:
            self.bert_tokenizer = None
            self.lowercase_input = False

    def _wordpiece_tokenize_input(self, tokens: List[str]) -> Tuple[List[str], List[int], List[int]]:
        """
        Convert a list of tokens to wordpiece tokens and offsets, as well as adding
        BERT CLS and SEP tokens to the begining and end of the sentence.

        A slight oddity with this function is that it also returns the wordpiece offsets
        corresponding to the _start_ of words as well as the end.

        We need both of these offsets (or at least, it's easiest to use both), because we need
        to convert the labels to tags using the end_offsets. However, when we are decoding a
        BIO sequence inside the SRL model itself, it's important that we use the start_offsets,
        because otherwise we might select an ill-formed BIO sequence from the BIO sequence on top of
        wordpieces (this happens in the case that a word is split into multiple word pieces,
        and then we take the last tag of the word, which might correspond to, e.g, I-V, which
        would not be allowed as it is not preceeded by a B tag).

        For example:

        `annotate` will be bert tokenized as ["anno", "##tate"].
        If this is tagged as [B-V, I-V] as it should be, we need to select the
        _first_ wordpiece label to be the label for the token, because otherwise
        we may end up with invalid tag sequences (we cannot start a new tag with an I).

        Returns
        -------
        wordpieces : List[str]
            The BERT wordpieces from the words in the sentence.
        end_offsets : List[int]
            Indices into wordpieces such that `[wordpieces[i] for i in end_offsets]`
            results in the end wordpiece of each word being chosen.
        start_offsets : List[int]
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
            start_offsets.append(cumulative + 1)
            cumulative += len(word_pieces)
            end_offsets.append(cumulative)
            word_piece_tokens.extend(word_pieces)

        wordpieces = ["[CLS]"] + word_piece_tokens + ["[SEP]"]

        return wordpieces, end_offsets, start_offsets

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        skipped_instances_count = 0
        instances_count = 0
        file_path = cached_path(file_path)
        logger.info("Reading SRL instances from dataset files at: %s", file_path)
        if self._domain_identifier is not None:
            logger.info("Filtering to only include file paths containing the %s domain", self._domain_identifier)


        if "bolt" in file_path:
            bolt_reader = Bolt()
            for sentence in self._bolt_subset(bolt_reader, file_path):
                tokens = [Token(t) for t in sentence.words]
                # if not sentence.srl_frames:
                #     # Sentence contains no predicates.
                #     tags = ["O" for _ in tokens]
                #     verb_label = [0 for _ in tokens]
                #     yield self.text_to_instance(tokens, verb_label, tags)
                # else:
                if sentence.srl_frames:
                    sense = [0 for _ in sentence.srl_frames]
                    for (_, tags) in sentence.srl_frames:
                        verb_indicator = [1 if label[-2:] == "-V" else 0 for label in tags]
                        yield self.text_to_instance(tokens, verb_indicator, tags, sense)
        elif "ontonotes" in file_path:
            ontonotes_reader = Ontonotes()
            for sentence in self._ontonotes_subset(ontonotes_reader, file_path, self._domain_identifier):
                tokens = [Token(t) for t in sentence.words]
                sense = sentence.word_senses
                #if not sentence.srl_frames:
                    # Sentence contains no predicates.
                    # tags = ["O" for _ in tokens]
                    # verb_label = [0 for _ in tokens]
                    # yield self.text_to_instance(tokens, verb_label, tags, None)
                if sentence.srl_frames:
                #else:
                    for (_, tags) in sentence.srl_frames:
                        verb_indicator = [1 if label[-2:] == "-V" else 0 for label in tags]
                        pred_index = verb_indicator.index(1)
                        sense = sentence.word_senses[pred_index]
                        if sense:
                            instances_count += 1
                            print('INSTANCE COUNT: ', instances_count)
                            yield self.text_to_instance(tokens, verb_indicator, tags, sense)
                        else:
                            skipped_instances_count += 1
                            print('SKIPPED COUNT: ', skipped_instances_count)
                else:
                    skipped_instances_count += 1
                    print('SKIPPED COUNT: ', skipped_instances_count)

    @staticmethod
    def _bolt_subset(bolt_reader: Bolt, file_path: str) -> Iterable[BoltSentence]:
        for conll_file in bolt_reader.dataset_path_iterator(file_path):
            yield from bolt_reader.sentence_iterator(conll_file)

    @staticmethod
    def _ontonotes_subset(ontonotes_reader: Ontonotes,
                          file_path: str,
                          domain_identifier: str) -> Iterable[OntonotesSentence]:
        """
        Iterates over the Ontonotes 5.0 dataset using an optional domain identifier.
        If the domain identifier is present, only examples which contain the domain
        identifier in the file path are yielded.
        """
        for conll_file in ontonotes_reader.dataset_path_iterator(file_path):
            if domain_identifier is None or f"/{domain_identifier}/" in conll_file:
                print('FILE: ', conll_file)
                yield from ontonotes_reader.sentence_iterator(conll_file)


    def text_to_instance(self,  # type: ignore
                         tokens: List[Token],
                         verb_label: List[int],
                         tags: List[str] = None,
                         sense: float = None) -> Instance:
        """
        We take `pre-tokenized` input here, along with a verb label.  The verb label should be a
        one-hot binary vector, the same length as the tokens, indicating the position of the verb
        to find arguments for.
        """
        # pylint: disable=arguments-differ
        metadata_dict: Dict[str, Any] = {}
        new_sentence, new_indices = separate_hyphens([t.text for t in tokens])
        new_verb_label = [0 for _ in new_sentence]
        for idx, indicator in enumerate(verb_label):
            if indicator == 1:
                for new_idx in new_indices[idx]:
                    new_verb_label[new_idx] = 1
        verb_label = new_verb_label

        if self.bert_tokenizer is not None:
            wordpieces, offsets, start_offsets = self._wordpiece_tokenize_input(new_sentence)
            new_verbs = _convert_verb_indices_to_wordpiece_indices(verb_label, offsets)
            metadata_dict["offsets"] = start_offsets
            # In order to override the indexing mechanism, we need to set the `text_id`
            # attribute directly. This causes the indexing to use this id.
            text_field = TextField([Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in wordpieces],
                                   token_indexers=self._token_indexers)
            verb_indicator = SequenceLabelField(new_verbs, text_field)

        else:
            text_field = TextField(tokens, token_indexers=self._token_indexers)
            verb_indicator = SequenceLabelField(verb_label, text_field)

        fields: Dict[str, Field] = {}
        fields['tokens'] = text_field
        fields['verb_indicator'] = verb_indicator

        if all([x == 0 for x in verb_label]):
            verb = None
            verb_index = None
        else:
            verb_index = verb_label.index(1)
            verb = tokens[verb_index].text


        metadata_dict["words"] = [x.text for x in tokens]
        metadata_dict["verb"] = verb
        metadata_dict["verb_index"] = verb_index

        if tags:
            new_tags = ['O' for _ in new_sentence]
            for idx, old_tag in enumerate(tags):
                for new_idx in new_indices[idx]:
                    new_tags[new_idx] = old_tag
            tags = new_tags
            if self.bert_tokenizer is not None:
                new_tags = _convert_tags_to_wordpiece_tags(tags, offsets)
                fields['tags'] = SequenceLabelField(new_tags, text_field)
            else:
                fields['tags'] = SequenceLabelField(tags, text_field)
            metadata_dict["gold_tags"] = tags
        
        if sense:
            fields['sense'] = LabelField(str(sense), label_namespace="sense_labels")

        fields["metadata"] = MetadataField(metadata_dict)
        return Instance(fields)
