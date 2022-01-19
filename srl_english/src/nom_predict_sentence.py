from typing import List, Iterator, Optional
import argparse
import sys
import json

from overrides import overrides

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import sanitize
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor, JsonDict
from allennlp.data import Instance

from nominal_srl.nominal_srl_predictor import NominalSemanticRoleLabelerPredictor

import predict_utils

desc = "Run nominal SRL predictor on a single sentence."

parser = argparse.ArgumentParser(description=desc)
parser.add_argument("archive_file", type=str, help="the archived model to make predictions with")
parser.add_argument('-s', '--input_sentence', type=str, help="the sentence to predict on", required=True)
parser.add_argument('-i', '--nom_indices', nargs='*', type=int, help="the indices of the nominal predicates", required=True)
parser.add_argument("--cuda_device", type=int, default=-1, help="id of GPU to use (if any)")
parser.add_argument('-o', '--output_file', type=str, default="output.txt", help="path to output file")
parser.add_argument('-ta', '--text_annotation', default=False, action='store_true', help="specify whether to produce the output in text annotation form")

args = parser.parse_args()


def _get_predictor(args) -> NominalSemanticRoleLabelerPredictor:
    check_for_gpu(args.cuda_device)
    archive = load_archive(
            args.archive_file,
            cuda_device=args.cuda_device,
            )
    return NominalSemanticRoleLabelerPredictor.from_archive(archive, "nombank-semantic-role-labeling")


class _PredictManager:
    def __init__(
        self,
        predictor: NominalSemanticRoleLabelerPredictor,
        input_sentence: str,
        indices: List[int],
        output_file: Optional[str],
        write_ta: bool,
    ) -> None:

        self._predictor = predictor
        self._indices = indices
        self._input_sentence = input_sentence
        if output_file is not None:
            self._output_file = open(output_file, "w")
        else:
            self._output_file = None
        self._write_ta = write_ta
        self.generator = "nominal_srl.nom_predict_sentence"

    def create_text_annotation(
        self, srl_output: JsonDict
    ) -> JsonDict:
        ta=  {"corpusId": "", "id": ""}
        tokens = srl_output.pop("words")
        text = self._input_sentence
        ta["text"] = text
        ta["tokens"] = tokens
        ta["tokenOffsets"] = predict_utils.create_token_char_offsets(text)
        sentence_end_positions = [i+1 for i,x in enumerate(tokens) if x=="."]
        sentences = {"generator": self.generator, "score": 1.0, "sentenceEndPositions": sentence_end_positions}
        ta["sentences"] = sentences

        # Create views.
        views = []
        views.append(predict_utils.create_sentence_view(tokens))
        views.append(predict_utils.create_tokens_view(tokens))
        views.append(self.create_srl_nom_view(srl_output.pop("nominals")))
        ta["views"] = views
        return sanitize(ta)

    def create_srl_nom_view(
        self, nom_srl_frames
    ) -> JsonDict:
        srl_nom_view = {"viewName": "SRL_NOM_NOMBANK"}
        constituents = []
        relations = []
        for frame in nom_srl_frames:
            predicate = frame.pop("nominal")
            description = frame.pop("description")
            tags = frame.pop("tags")
            predicate_idx = frame.pop("predicate_index")
            properties = {"SenseNumber": "NA", "predicate": predicate}
            if len(predicate_idx)>1:
                print('Multiple indices of predicate. Using first.')
            constituent = {"label": "Predicate", "score": 1.0, "start": predicate_idx[0], "end": predicate_idx[0]+1, "properties": properties}
            predicate_constituent_idx = len(constituents)
            constituents.append(constituent)
            active_tag = ""
            active_tag_start_idx = -1
            for tag_idx, tag in enumerate(tags):
                if tag in {"O", "B-V"}:
                    if active_tag != "":
                        constituent = {"label": active_tag, "score": 1.0, "start": active_tag_start_idx, "end": tag_idx}
                        relation = {"relationName": active_tag, "srcConstituent": predicate_constituent_idx, "targetConstituent": len(constituents)}
                        relations.append(relation)
                        constituents.append(constituent)
                        active_tag = ""
                        active_tag_start_idx = -1
                    continue
                if tag[2:] == active_tag:
                    continue
                else:
                    if active_tag != "":
                        constituent = {"label": active_tag, "score": 1.0, "start": active_tag_start_idx, "end": tag_idx}
                        relation = {"relationName": active_tag, "srcConstituent": predicate_constituent_idx, "targetContituent": len(constituents)}
                        relations.append(relation)
                        constituents.append(constituent)
                    active_tag = tag[2:]
                    active_tag_start_idx = tag_idx
        nom_view_data = [{"viewType": "", "viewName": "SRL_NOM_NOMBANK", "generator": self.generator, "score": 1.0, "constituents": constituents, "relations": relations}]
        srl_nom_view["viewData"] = nom_view_data
        return srl_nom_view

    def _print_to_file(
        self, prediction: str
    ) -> None:
        if self._output_file is not None:
            self._output_file.write(prediction)
            self._output_file.close()
        else:
            print("No output file was specified. Writing to STDOUT instead.")
            print(prediction)

    def run(self) -> None:
        result = self._predictor.predict(self._input_sentence, self._indices)
        print('OUTPUT_DICT: ', result)
        if self._write_ta:
            ta = self.create_text_annotation(result)
            self._print_to_file(json.dumps(ta, indent=4))
        else:
            self._print_to_file(json.dumps(result, indent=4))

predictor = _get_predictor(args)

manager = _PredictManager(
    predictor,
    args.input_sentence,
    args.nom_indices,
    args.output_file,
    args.text_annotation,
    )    

manager.run()
