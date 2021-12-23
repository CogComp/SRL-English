import json
import os
import sys

from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import JsonDict
from id_nominal.nominal_predictor import NominalIdPredictor
from nominal_sense_srl.predictor import NomSenseSRLPredictor
from nominal_sense_srl.predictor_all import AllNomSenseSRLPredictor
from verb_sense_srl.predictor import SenseSRLPredictor
from prep_srl.preposition_srl_predictor import PrepositionSemanticRoleLabelerPredictor
import prep_srl.preposition_srl_reader
import prep_srl.preposition_srl_model
from tabular_view import *

mode = sys.argv[1]
content = sys.argv[2]
filename = sys.argv[3]
input_directory = sys.argv[4]
output_directory = sys.argv[5]



nom_sense_srl_archive = load_archive('/shared/celinel/test_allennlp/v0.9.0/nom-sense-srl/model.tar.gz',)
verb_sense_srl_archive = load_archive('/shared/celinel/test_allennlp/v0.9.0/verb-sense-srl/model.tar.gz',)
nom_sense_srl_predictor = NomSenseSRLPredictor.from_archive(nom_sense_srl_archive, "nombank-sense-srl")
all_nom_sense_srl_predictor = AllNomSenseSRLPredictor.from_archive(nom_sense_srl_archive, "all-nombank-sense-srl")
verb_sense_srl_predictor = SenseSRLPredictor.from_archive(verb_sense_srl_archive, "sense-semantic-role-labeling")
print('LOADED VERB MODEL')
nom_id_archive = load_archive('/shared/celinel/test_allennlp/v0.9.0/test-id-bert/model.tar.gz',)
nom_id_predictor = NominalIdPredictor.from_archive(nom_id_archive, "nombank-id")
print('LOADED NOM MODEL')
prep_srl_archive = load_archive("/shared/fmarini/preposition-SRL/preposition-SRL/new-srl-manual/model.tar.gz",)
prep_srl_predictor = PrepositionSemanticRoleLabelerPredictor.from_archive(prep_srl_archive, "preposition-semantic-role-labeling")
print('LOADED PREP MODEL')



def separate_hyphens(og_sentence):
    new_sentence = []
    i = 0
    for word in og_sentence:
        h_idx = word.find('-')
        bslash_idx = word.find('/')
        h_bs_idx = min(h_idx, bslash_idx) if h_idx>=0 and bslash_idx>=0 else max(h_idx, bslash_idx)
        prev_h_bs_idx = -1
        while h_bs_idx > 0:
            subsection = word[prev_h_bs_idx+1:h_bs_idx]
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
            i += 1
    return new_sentence

class Annotation(object):    
    def _convert_id_to_srl_input(self, id_output):
        indices = [idx for idx in range(len(id_output["nominals"])) if id_output["nominals"][idx]==1]
        shiftleft = 0
        new_indices = []
        new_tokens = []
        for idx, token in enumerate(id_output["words"]):
            if token=="" or token.isspace():
                shiftleft += 1
            else:
                if idx in indices:
                    new_indices.append(idx-shiftleft)
                new_tokens.append(token)
        srl_input = {"sentence": " ".join(new_tokens), "indices": new_indices}
        return srl_input
    
    
    def annotate(self, sentence=None):
        sentence = separate_hyphens(sentence.split())
        input_json_data = {"sentence": " ".join(sentence)}

        id_output = nom_id_predictor.predict_json(input_json_data)
        nom_srl_input = self._convert_id_to_srl_input(id_output)
        nom_srl_output = nom_sense_srl_predictor.predict_json(nom_srl_input)
        all_nom_srl_output = all_nom_sense_srl_predictor.predict_json(input_json_data)
        verb_srl_output = verb_sense_srl_predictor.predict_json(input_json_data)
        prep_srl_output = prep_srl_predictor.predict_json(input_json_data)
        tabular_structure = TabularView()
        tabular_structure.update_sentence(nom_srl_output)
        tabular_structure.update_view("SRL_VERB", verb_srl_output)
        tabular_structure.update_view("SRL_NOM", nom_srl_output)
        tabular_structure.update_view("SRL_NOM_ALL", all_nom_srl_output)
        tabular_structure.update_view("SRL_PREP", prep_srl_output)
        return tabular_structure.get_textannotation()

    def annotateMain(self, mode = "content", content="", filename="", input_directory="", output_directory=""):
        if mode == "content":
            output = self.annotate(sentence=content)
            print(output)
            return output
        elif mode == "file":
            content = open(filename, "r").read()
            output = self.annotate(content)
            print(output)
            return output
        
        elif mode == "directory":
            file_list = os.listdir(input_directory)
            for filename in file_list:
                content = open(input_directory + "/" + filename, "r").read()
                annJson =  self.annotate(content)
                print("_" * 80)
                print(filename)
                print("_" * 80)
                print(annJson)
                print("_" * 50)


if __name__ == '__main__':
    annotationObj = Annotation()
    annotationObj.annotateMain(mode = mode, content=content, filename=filename, input_directory=input_directory, output_directory=output_directory)
