import logging
import os
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

def get_bio_tags(args: List[str], new_indices: List[List[int]], new_sentence: List[str]):
    all_args_ordered = []
    for arg in args:
        subargs = arg[:arg.find('-')]
        pre = ""
        if '*' in subargs:
            pre = "R-"
            subargs = subargs.split("*")
        elif ',' in subargs:
            pre = "C-"
            subargs = subargs.split(",")
        else:
            subargs = [subargs]
        label = arg[arg.find('-')+1:]

        isfirst = True
        for arg in subargs:
            # If is a hyphenation for a span of words, include all inside up to edges' hyphens, if existant. Assumes continuity.
            start = arg[:arg.find(':')]
            sub_idx = start.find('_')
            if sub_idx < 0:
                new_start = new_indices[int(start)][0]
            else:
                if len(new_indices[int(start[:sub_idx])]) <= 1:
                    # Specified hyphenated arg, but start index not actually a hyphenation. Consider moving handling of this to the span.srl generating code.
                    new_start = new_indices[int(start[:sub_idx])][0]
                elif int(start[sub_idx+1:]) >= len(new_indices[int(start[:sub_idx])]):
                    print("Faulty data point with arg ", arg)
                    continue
                else:
                    new_start = new_indices[int(start[:sub_idx])][int(start[sub_idx+1:])]
            end = arg[arg.find(':')+1:]
            sub_idx = end.find('_')
            if sub_idx < 0:
                new_end = new_indices[int(end)][0]
            else:
                if len(new_indices[int(end[:sub_idx])]) <= 1:
                    new_end = new_indices[int(end[:sub_idx])][0]
                elif int(end[sub_idx+1:]) >= len(new_indices[int(end[:sub_idx])]):
                    print("Faulty data point with arg ", arg)
                    continue
                else:
                    new_end = new_indices[int(end[:sub_idx])][int(end[sub_idx+1:])]
            if isfirst:
                all_args_ordered.append((new_start, new_end, label, ""))
            else:
                all_args_ordered.append((new_start, new_end, label, pre))
            isfirst = False
    all_args_ordered = sorted(all_args_ordered, key=lambda x: x[0])

    bio_tags = ['O' for _ in range(len(new_sentence))]
    for arg in all_args_ordered:
        current_label_at_start = bio_tags[arg[0]]
        current_label_at_end = bio_tags[arg[1]]
        if current_label_at_start != 'O':
            if current_label_at_end != 'O':
                if current_label_at_start[2:] == current_label_at_end[2:]:
                    # This span is dominated, so we can just skip it.
                    continue
            else:
                if current_label_at_start[2:4] in {"R-", "C-"}:
                    if current_label_at_start[:2] == "I-":
                        continue
                else:
                    continue
                if arg[2][:2] in {"R-", "C-"}:
                    continue
        bio_tags[arg[0]] = "B-{0}{1}".format(arg[3],arg[2]) ##separate
        # bio_tags[arg[0]] = "B-NOM-{0}{1}".format(arg[3],arg[2]) ##separate
        i = arg[0]+1
        while i <= arg[1]:
            bio_tags[i] = "I-{0}{1}".format(arg[3],arg[2]) ##separate
            # bio_tags[i] = "I-NOM-{0}{1}".format(arg[3],arg[2]) ##separate
            i += 1

    return bio_tags

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


def _convert_indices_to_wordpiece_indices(indices: List[int], offsets: List[int]): # pylint: disable=invalid-name
    """
    Converts binary indicators to account for a wordpiece tokenizer,
    extending/modifying BIO tags where appropriate to deal with words which
    are split into multiple wordpieces by the tokenizer.

    This is only used if you pass a `bert_model_name` to the dataset reader below.

    Parameters
    ----------
    indices : `List[int]`
        The binary indicators, 0 for not a predicate, 1 for predicate.
    offsets : `List[int]`
        The wordpiece offsets.

    Returns
    -------
    The new indices.
    """
    j = 0
    new_indices = []
    for i, offset in enumerate(offsets):
        indicator = indices[i]
        while j < offset:
            new_indices.append(indicator)
            j += 1

    # Add 0 indicators for cls and sep tokens.
    return [0] + new_indices + [0]


@DatasetReader.register("combined-srl")
class CombinedSRLReader(DatasetReader):
    """
    This DatasetReader is designed to read in the English Ontonotes, Bolt, and Nombank data
    for semantic role labelling. 
    It returns a dataset of instances with the following fields:

    tokens : ``TextField``
        The tokens in the sentence.
    predicate_indicator : ``SequenceLabelField``
        A sequence of binary indicators for whether the word is the predicate for this frame.
    tags : ``SequenceLabelField``
        A sequence of Propbank tags for the given predicate in a BIO format.

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
        and will expand your tags and predicate indicators accordingly. If not,
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
        
        # file_path = cached_path(file_path)
        logger.info("Reading SRL instances from dataset files at: %s", file_path)
        # READ NOMBANK        
        nombank_data_file = os.path.join(file_path, "nombank.srl")            
        srl_data = self.read_nom_srl(nombank_data_file)
        for (og_nom_loc, new_indices, new_sentence, new_tags) in srl_data:
            new_tokens = [Token(t) for t in new_sentence]
            new_pred_idx = new_indices[og_nom_loc[0]]
            if len(og_nom_loc) > 1:
                if og_nom_loc[1] >= len(new_pred_idx):
                    print('Faulty data point. Trying to access hyphenation that does not exist.')
                    continue
                new_pred_idx = [new_pred_idx[og_nom_loc[1]]]
            nom_indicator = [1 if i in new_pred_idx else 0 for i in range(len(new_tokens))]
            yield self.text_to_instance(new_tokens, nom_indicator, new_tags)
        # READ BOLT
        bolt_file_path = os.path.join(file_path, "bolt_df")
        bolt_reader = Bolt()
        for sentence in self._bolt_subset(bolt_reader, bolt_file_path):
            tokens = [Token(t) for t in sentence.words]
            if not sentence.srl_frames:
                # Sentence contains no predicates.
                tags = ["O" for _ in tokens]
                verb_label = [0 for _ in tokens]
                yield self.text_to_instance(tokens, verb_label, tags)
            else:
                for (_, tags) in sentence.srl_frames:
                    # TODO consider for the separate tags, should we separate them by POS actually instead of by dataset?
                    verb_indicator = [1 if label[-2:] == "-V" else 0 for label in tags]
                    # pred_idx = verb_indicator.index(1)
                    # if sentence.pos_tags[pred_idx].startswith("JJ"):
                    yield self.text_to_instance(tokens, verb_indicator, tags)
        # READ ONTONOTES    
        ontonotes_file_path = os.path.join(file_path, "conll-formatted-ontonotes-5.0")
        ontonotes_reader = Ontonotes()
        for sentence in self._ontonotes_subset(ontonotes_reader, ontonotes_file_path):
            tokens=[Token(t) for t in sentence.words]
            if not sentence.srl_frames:
                tags = ["O" for _ in tokens]
                verb_label = [0 for _ in tokens]
                yield self.text_to_instance(tokens, verb_label, tags)
            else:
                for (_, tags) in sentence.srl_frames:
                    verb_indicator = [1 if label[-2:]=="-V" else 0 for label in tags]
                    yield self.text_to_instance(tokens, verb_indicator, tags)
        # READ PREPOSITION
        prep_file_path = os.path.join(file_path, "preposition.txt")
        f = open(prep_file_path, "r")
        for line in f.readlines():
            str_list = line.strip().split()
            separator_index = str_list.index("|||")
            filename = str_list[0]
            sentence = str_list[1:separator_index]
            # supersense1 = str_list[separator_index+1]
            # supersense2 = str_list[separator_index+2]
            old_tags = str_list[separator_index+3:]
            tags = []
            for i, tag in enumerate(old_tags):
                # lines ending in ##shared are used in the shared label task
                # lines ending in ##separate are used in the separate label task
                # if tag[2:] == "GOV": ##shared
                #     tags.append("{}ARG0".format(tag[:2])) ##shared
                # elif tag[2:] == "OBJ": ##shared
                #     tags.append("{}ARG1".format(tag[:2])) ##shared
                # else: ##shared
                #     tags.append(tag) ##shared
                # tags.append("{}PREP-{}".format(tag[:2],tag[2:])) ##separate
                tags.append(tag)
            new_tokens = [Token(t) for t in sentence]
            # predicate_location = str_list.index("B-PREP")
            predicate_indicator = [1 if "PREP" in tag else 0 for tag in tags]
            assert len(predicate_indicator) == len(tags)
            yield self.text_to_instance(new_tokens, predicate_indicator, tags)
        f.close()

    @staticmethod
    def _ontonotes_subset(ontonotes_reader: Ontonotes,
                          file_path: str) -> Iterable[OntonotesSentence]:
        for conll_file in ontonotes_reader.dataset_path_iterator(file_path):
            yield from ontonotes_reader.sentence_iterator(conll_file)

    @staticmethod
    def _bolt_subset(bolt_reader: Bolt,
                     file_path: str) -> Iterable[BoltSentence]:
        """
        Iterates over the Ontonotes 5.0 dataset using an optional domain identifier.
        If the domain identifier is present, only examples which contain the domain
        identifier in the file path are yielded.
        """
        for conll_file in bolt_reader.dataset_path_iterator(file_path):
            yield from bolt_reader.sentence_iterator(conll_file)

    def read_nom_srl(self, filename):
        """
        This process reads in the nominal srl data in span format, and 
        converts it to BIO format. 

        example input line:
        wsj/05/wsj_0544.mrg 0 Air & Water Technologies Corp. completed the acquisition of Falcon Associates Inc. , a Bristol , Pa. , asbestos-abatement concern , for $ 25 million of stock . ||| 01 18_1:18_1-rel 18_0:18_0-ARG1

        its output:
        (
            og_sentence = ['Air', '&', 'Water', 'Technologies' ... 'asbestos-abatement', 'concern', ',', 'for', '$', '25', 'million', 'of', 'stock', '.'],
            og_nom_loc = (18, 1),
            new_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, [18, 19], 20, 21, ... , 28],
            new_sentence = ['Air', '&', ... 'asbestos-', 'abatement', 'concern', ...],
            new_tags = ['O', ... 'O', 'B-ARG1', 'B-REL', 'O', ..., 'O']
            
        )
        """

        fin = open(filename, 'r')
        data = []
        for line in fin.readlines():
            str_list = line.strip().split()
            separator_index = str_list.index("|||")
            og_sentence = str_list[2:separator_index]
            args = str_list[separator_index+2:]
            # Get index of predicate. Predicate is always first argument.
            predicate_loc = args[0][:args[0].find(':')]
            sub_idx = predicate_loc.find('_')
            if sub_idx < 0:
                og_nom_loc = [int(predicate_loc)]
            else:
                og_nom_loc = (int(predicate_loc[:sub_idx]), int(predicate_loc[sub_idx+1:]))
            # Get new indices and new sentence, hyphenations separated.
            new_sentence, new_indices = separate_hyphens(og_sentence)
            # Get BIO tags from argument spans. 
            new_tags = get_bio_tags(args[1:], new_indices, new_sentence)
            assert len(new_tags) == len(new_sentence)
            data.append((og_nom_loc, new_indices, new_sentence, new_tags))
        fin.close()
        return data

    def text_to_instance(self,  # type: ignore
                         tokens: List[Token],
                         predicate_label: List[int],
                         tags: List[str] = None) -> Instance:
        """
        We take `pre-tokenized` input here, along with a predicate label.  The predicate label should be a
        one-hot binary vector, the same length as the tokens, indicating the position of the predicate
        to find arguments for.
        """
        # print('TEXT_TO_INSTANCE: ', tokens, ', ', predicate_label, ', ', tags, ', ', sense)
        # pylint: disable=arguments-differ
        metadata_dict: Dict[str, Any] = {}
        if self.bert_tokenizer is not None:
            wordpieces, offsets, start_offsets = self._wordpiece_tokenize_input([t.text for t in tokens])
            new_predicates = _convert_indices_to_wordpiece_indices(predicate_label, offsets)
            metadata_dict["offsets"] = start_offsets
            # In order to override the indexing mechanism, we need to set the `text_id`
            # attribute directly. This causes the indexing to use this id.
            text_field = TextField([Token(t, text_id=self.bert_tokenizer.vocab[t]) for t in wordpieces],
                                   token_indexers=self._token_indexers)
            predicate_indicator = SequenceLabelField(new_predicates, text_field)

        else:
            text_field = TextField(tokens, token_indexers=self._token_indexers)
            predicate_indicator = SequenceLabelField(predicate_label, text_field)

        fields: Dict[str, Field] = {}
        fields['tokens'] = text_field
        fields['predicate_indicator'] = predicate_indicator

        if all([x == 0 for x in predicate_label]):
            predicate = None
            predicate_index = None
        else:
            # predicate_index = predicate_label.index(1)
            # predicate = tokens[predicate_index].text

            predicate_index = [i for i in range(len(predicate_label)) if predicate_label[i]==1]
            predicate = ''
            for p_idx in predicate_index:
                predicate += tokens[p_idx].text


        metadata_dict["words"] = [x.text for x in tokens]
        metadata_dict["predicate"] = predicate
        metadata_dict["predicate_index"] = predicate_index

        if tags:
            if self.bert_tokenizer is not None:
                new_tags = _convert_tags_to_wordpiece_tags(tags, offsets)
                fields['tags'] = SequenceLabelField(new_tags, text_field)
            else:
                fields['tags'] = SequenceLabelField(tags, text_field)
            metadata_dict["gold_tags"] = tags

        fields["metadata"] = MetadataField(metadata_dict)
        return Instance(fields)
