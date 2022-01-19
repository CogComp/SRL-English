from typing import DefaultDict, List, Optional, Iterator, Set, Tuple
from collections import defaultdict
import codecs
import os
import logging

from nltk import Tree

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

TypedSpan = Tuple[int, Tuple[int, int]]  # pylint: disable=invalid-name
TypedStringSpan = Tuple[str, Tuple[int, int]]  # pylint: disable=invalid-name

class BoltSentence:
    """
    A class representing the annotations available for a single CONLL formatted sentence.

    Parameters
    ----------
    document_id : ``str``
        This is a variation on the document filename
    sentence_id : ``int``
        The integer ID of the sentence within a document.
    words : ``List[str]``
        This is the tokens as segmented/tokenized in the Treebank.
    pos_tags : ``List[str]``
        This is the Penn-Treebank-style part of speech. When parse information is missing,
        all parts of speech except the one for which there is some sense or proposition
        annotation are marked with a XX tag. The verb is marked with just a VERB tag.
    parse_tree : ``nltk.Tree``
        An nltk Tree representing the parse. It includes POS tags as pre-terminal nodes.
        When the parse information is missing, the parse will be ``None``.
    predicate_lemmas : ``List[Optional[str]]``
        The predicate lemma of the words for which we have semantic role
        information or word sense information. All other indices are ``None``.
    predicate_framenet_ids : ``List[Optional[int]]``
        The PropBank frameset ID of the lemmas in ``predicate_lemmas``, or ``None``.
    srl_frames : ``List[Tuple[str, List[str]]]``
        A dictionary keyed by the verb in the sentence for the given
        Propbank frame labels, in a BIO format.
    """
    def __init__(self,
                 document_id: str,
                 sentence_id: int,
                 words: List[str],
                 pos_tags: List[str],
                 parse_tree: Optional[Tree],
                 predicate_lemmas: List[Optional[str]],
                 predicate_framenet_ids: List[Optional[str]],
                 srl_frames: List[Tuple[str, List[str]]]) -> None:

        self.document_id = document_id
        self.sentence_id = sentence_id
        self.words = words
        self.pos_tags = pos_tags
        self.parse_tree = parse_tree
        self.predicate_lemmas = predicate_lemmas
        self.predicate_framenet_ids = predicate_framenet_ids
        self.srl_frames = srl_frames


class Bolt:
    """
    This DatasetReader is designed to read in the Bolt data
    in the format used by the CoNLL 2011/2012 shared tasks.

    The file path provided to this class can then be any of the train, test or development
    directories(or the top level data directory, if you are not utilizing the splits).

    The data has the following format, ordered by column.

    1 Document ID : ``str``
        This is a variation on the document filename
    2 Part number : ``int``
        Some files are divided into multiple parts numbered as 000, 001, 002, ... etc.
    3 Word number : ``int``
        This is the word index of the word in that sentence.
    4 Word : ``str``
        This is the token as segmented/tokenized in the Treebank. Initially the ``*_skel`` file
        contain the placeholder [WORD] which gets replaced by the actual token from the
        Treebank which is part of the OntoNotes release.
    5 POS Tag : ``str``
        This is the Penn Treebank style part of speech. When parse information is missing,
        all part of speeches except the one for which there is some sense or proposition
        annotation are marked with a XX tag. The verb is marked with just a VERB tag.
    6 Parse bit: ``str``
        This is the bracketed structure broken before the first open parenthesis in the parse,
        and the word/part-of-speech leaf replaced with a ``*``. When the parse information is
        missing, the first word of a sentence is tagged as ``(TOP*`` and the last word is tagged
        as ``*)`` and all intermediate words are tagged with a ``*``.
    7 Predicate lemma: ``str``
        The predicate lemma is mentioned for the rows for which we have semantic role
        information or word sense information. All other rows are marked with a "-".
    8 Predicate Frameset ID: ``int``
        The PropBank frameset ID of the predicate in Column 7.
    9+ Predicate Arguments: ``str``
        There is one column each of predicate argument structure information for the predicate
        mentioned in Column 7. If there are no predicates tagged in a sentence this is a
        single column with all rows marked with an ``*``.
    """
    def dataset_iterator(self, file_path: str) -> Iterator[BoltSentence]:
        """
        An iterator over the entire dataset, yielding all sentences processed.
        """
        for conll_file in self.dataset_path_iterator(file_path):
            yield from self.sentence_iterator(conll_file)

    @staticmethod
    def dataset_path_iterator(file_path: str) -> Iterator[str]:
        """
        An iterator returning file_paths in a directory
        containing CONLL-formatted files.
        """
        logger.info("Reading CONLL sentences from dataset files at: %s", file_path)
        for root, _, files in list(os.walk(file_path)):
            for data_file in files:
                # These are a relic of the dataset pre-processing. Every
                # file will be duplicated - one file called filename.gold_skel
                # and one generated from the preprocessing called filename.gold_conll.
                if not data_file.endswith("gold_conll"):
                    continue
                yield os.path.join(root, data_file)

    def dataset_document_iterator(self, file_path: str) -> Iterator[List[BoltSentence]]:
        """
        An iterator over CONLL formatted files which yields documents, regardless
        of the number of document annotations in a particular file. This is useful
        for conll data which has been preprocessed, such as the preprocessing which
        takes place for the 2012 CONLL Coreference Resolution task.
        """
        with codecs.open(file_path, 'r', encoding='utf8') as open_file:
            conll_rows = []
            document: List[BoltSentence] = []
            for line in open_file:
                line = line.strip()
                if line != '' and not line.startswith('#'):
                    # Non-empty line. Collect the annotation.
                    conll_rows.append(line)
                else:
                    if conll_rows:
                        document.append(self._conll_rows_to_sentence(conll_rows))
                        conll_rows = []
                if line.startswith("#end document"):
                    yield document
                    document = []
            if document:
                # Collect any stragglers or files which might not
                # have the '#end document' format for the end of the file.
                yield document

    def sentence_iterator(self, file_path: str) -> Iterator[BoltSentence]:
        """
        An iterator over the sentences in an individual CONLL formatted file.
        """
        for document in self.dataset_document_iterator(file_path):
            for sentence in document:
                yield sentence

    def _conll_rows_to_sentence(self, conll_rows: List[str]) -> BoltSentence:
        document_id: str = None
        sentence_id: int = None
        # The words in the sentence.
        sentence: List[str] = []
        # The pos tags of the words in the sentence.
        pos_tags: List[str] = []
        # the pieces of the parse tree.
        parse_pieces: List[str] = []
        # The lemmatised form of the words in the sentence which
        # have SRL or word sense information.
        predicate_lemmas: List[str] = []
        # The FrameNet ID of the predicate.
        predicate_framenet_ids: List[str] = []

        verbal_predicates: List[str] = []
        span_labels: List[List[str]] = []
        current_span_labels: List[str] = []

        for index, row in enumerate(conll_rows):
            conll_components = row.split()
            document_id = conll_components[0]
            sentence_id = int(conll_components[1])
            word = conll_components[3]
            pos_tag = conll_components[4]
            parse_piece = conll_components[5]

            # Replace brackets in text and pos tags
            # with a different token for parse trees.
            if pos_tag != "XX" and word != "XX":
                if word == "(":
                    parse_word = "-LRB-"
                elif word == ")":
                    parse_word = "-RRB-"
                else:
                    parse_word = word
                if pos_tag == '(':
                    pos_tag = '-LRB-'
                if pos_tag == ')':
                    pos_tag = '-RRB-'
                (left_brackets, right_hand_side) = parse_piece.split('*')
                # only keep ')' if there are nested brackets with nothing in them.
                right_brackets = right_hand_side.count(')') * ')'
                parse_piece = f'{left_brackets} ({pos_tag} {parse_word}) {right_brackets}'
            else:
                # There are some bad annotations in the CONLL data.
                # They contain no information, so to make this explicit,
                # we just set the parse piece to be None which will result
                # in the overall parse tree being None.
                parse_piece = None

            lemmatised_word = conll_components[6]
            framenet_id = conll_components[7]

            if not span_labels:
                # If this is the first word in the sentence, create
                # empty lists to collect the NER and SRL BIO labels.
                # We can't do this upfront, because we don't know how many
                # components we are collecting, as a sentence can have
                # variable numbers of SRL frames.
                span_labels = [[] for _ in conll_components[8:]]
                # Create variables representing the current label for each label
                # sequence we are collecting.
                current_span_labels = [None for _ in conll_components[8:]]

            self._process_span_annotations_for_word(conll_components[8:],
                                                    span_labels,
                                                    current_span_labels)

            # If any annotation marks this word as a verb predicate,
            # we need to record its index. This also has the side effect
            # of ordering the verbal predicates by their location in the
            # sentence, automatically aligning them with the annotations.
            word_is_verbal_predicate = any(["(V" in x for x in conll_components[8:]])
            if word_is_verbal_predicate:
                verbal_predicates.append(word)

            sentence.append(word)
            pos_tags.append(pos_tag)
            parse_pieces.append(parse_piece)
            predicate_lemmas.append(lemmatised_word if lemmatised_word != "-" else None)
            predicate_framenet_ids.append(framenet_id if framenet_id != "-" else None)

        srl_frames = [(predicate, labels) for predicate, labels
                      in zip(verbal_predicates, span_labels)]

        if all(parse_pieces):
            parse_tree = Tree.fromstring("".join(parse_pieces))
        else:
            parse_tree = None
        return BoltSentence(document_id,
                                 sentence_id,
                                 sentence,
                                 pos_tags,
                                 parse_tree,
                                 predicate_lemmas,
                                 predicate_framenet_ids,
                                 srl_frames)

    @staticmethod
    def _process_span_annotations_for_word(annotations: List[str],
                                           span_labels: List[List[str]],
                                           current_span_labels: List[Optional[str]]) -> None:
        """
        Given a sequence of different label types for a single word and the current
        span label we are inside, compute the BIO tag for each label and append to a list.

        Parameters
        ----------
        annotations: ``List[str]``
            A list of labels to compute BIO tags for.
        span_labels : ``List[List[str]]``
            A list of lists, one for each annotation, to incrementally collect
            the BIO tags for a sequence.
        current_span_labels : ``List[Optional[str]]``
            The currently open span per annotation type, or ``None`` if there is no open span.
        """
        for annotation_index, annotation in enumerate(annotations):
            # strip all bracketing information to
            # get the actual propbank label.
            label = annotation.strip("()*")

            if "(" in annotation:
                # Entering into a span for a particular semantic role label.
                # We append the label and set the current span for this annotation.
                bio_label = "B-" + label
                span_labels[annotation_index].append(bio_label)
                current_span_labels[annotation_index] = label
            elif current_span_labels[annotation_index] is not None:
                # If there's no '(' token, but the current_span_label is not None,
                # then we are inside a span.
                bio_label = "I-" + current_span_labels[annotation_index]
                span_labels[annotation_index].append(bio_label)
            else:
                # We're outside a span.
                span_labels[annotation_index].append("O")
            # Exiting a span, so we reset the current span label for this annotation.
            if ")" in annotation:
                current_span_labels[annotation_index] = None
