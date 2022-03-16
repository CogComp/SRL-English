import cherrypy
import json
import os
import sys

from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import JsonDict
from id_nominal.nominal_predictor import NominalIdPredictor
# from nominal_srl.nominal_srl_predictor import NominalSemanticRoleLabelerPredictor
# from nominal_srl.nominal_srl_predictor_all import NounSemanticRoleLabelerPredictor
from nominal_sense_srl.predictor import NomSenseSRLPredictor
from nominal_sense_srl.predictor_all import AllNomSenseSRLPredictor
# from allennlp.predictors.semantic_role_labeler import SemanticRoleLabelerPredictor as VerbSemanticRoleLabelerPredictor
from verb_sense_srl.predictor import SenseSRLPredictor
from prep_srl.preposition_srl_predictor import PrepositionSemanticRoleLabelerPredictor
import prep_srl.preposition_srl_reader
import prep_srl.preposition_srl_model
from tabular_view import *
import torch

serverURL = sys.argv[1]
serverPort = int( sys.argv[2] )

cuda_available = torch.cuda.is_available()

nom_sense_srl_archive = load_archive('/shared/celinel/test_allennlp/v0.9.0/nom-sense-srl/model.tar.gz',)
verb_sense_srl_archive = load_archive('/shared/celinel/test_allennlp/v0.9.0/verb-sense-srl/model.tar.gz',)
nom_sense_srl_predictor = NomSenseSRLPredictor.from_archive(nom_sense_srl_archive, "nombank-sense-srl")
all_nom_sense_srl_predictor = AllNomSenseSRLPredictor.from_archive(nom_sense_srl_archive, "all-nombank-sense-srl")
verb_sense_srl_predictor = SenseSRLPredictor.from_archive(verb_sense_srl_archive, "sense-semantic-role-labeling")
print('LOADED VERB MODEL')
# verb_srl_archive = load_archive('/shared/celinel/test_allennlp/v0.9.0/verb-srl-bert/model.tar.gz',)
# verb_srl_predictor = VerbSemanticRoleLabelerPredictor.from_archive(verb_srl_archive, "semantic-role-labeling")
nom_id_archive = load_archive('/shared/celinel/test_allennlp/v0.9.0/test-id-bert/model.tar.gz',)
nom_id_predictor = NominalIdPredictor.from_archive(nom_id_archive, "nombank-id")
# nom_srl_archive = load_archive('/shared/celinel/test_allennlp/v0.9.0/nom-srl-bert/model.tar.gz',)
# nom_srl_predictor = NominalSemanticRoleLabelerPredictor.from_archive(nom_srl_archive, "nombank-semantic-role-labeling")
# noun_srl_predictor = NounSemanticRoleLabelerPredictor.from_archive(nom_srl_archive, "noun-semantic-role-labeling")
print('LOADED NOM MODEL')
prep_srl_archive = load_archive("/shared/fmarini/preposition-SRL/preposition-SRL/new-srl-manual/model.tar.gz",)
prep_srl_predictor = PrepositionSemanticRoleLabelerPredictor.from_archive(prep_srl_archive, "preposition-semantic-role-labeling")
print('LOADED PREP MODEL')

if cuda_available:
    nom_sense_srl_predictor._model = nom_sense_srl_predictor._model.cuda()
    all_nom_sense_srl_predictor._model = all_nom_sense_srl_predictor._model.cuda()
    verb_sense_srl_predictor._model = verb_sense_srl_predictor._model.cuda()
    nom_id_predictor._model = nom_id_predictor._model.cuda()
    prep_srl_predictor._model = prep_srl_predictor._model.cuda()


def separate_hyphens(og_sentence):
    new_sentence = []
    i = 0
    for word in og_sentence:
        h_idx = word.find('-')
        bslash_idx = word.find('/')
        h_bs_idx = min(h_idx, bslash_idx) if h_idx>=0 and bslash_idx>=0 else max(h_idx, bslash_idx)
        prev_h_bs_idx = -1
        while h_bs_idx > 0:
            # subsection = word[prev_h_bs_idx+1:h_bs_idx+1]
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

class MyWebService(object):
    
    # global tabular_structure
    # tabular_structure = TabularView()

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


    @cherrypy.expose
    def index(self):
        return open('public/srl.html')

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def info(self, **params):
        return {"status":"online"}

    @cherrypy.expose
    def halt(self, **params):
        cherrypy.engine.exit()

    @cherrypy.expose
    @cherrypy.tools.json_in()
    @cherrypy.tools.json_out()
    def annotate(self, sentence=None):
        try:
            input_json_data = cherrypy.request.json
            input_json_data["sentence"] = " ".join(separate_hyphens(input_json_data["sentence"].split()))
        except:
            # data = cherrypy.request.params
            if sentence is None:
                cherrypy.response.headers['Content-Type'] = 'text/plain'
                input_data = cherrypy.request.body.readline()
                sentence = input_data.decode("utf-8")
                input_json_data = {"sentence": " ".join(separate_hyphens(sentence.split()))}
            else:
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

if __name__ == '__main__':
    conf = {
        '/': {
            'tools.sessions.on': True,
            'tools.staticdir.root': os.path.abspath(os.getcwd())
        },
    }
    print("Starting rest service...")
    '''
    config = {'server.socket_host': '0.0.0.0'}
    cherrypy.config.update(config)
    cherrypy.config.update({'server.socket_port': 8043})
    cherrypy.quickstart(MyWebService(), '/', conf)
    '''
    config = {'server.socket_host': serverURL}
    cherrypy.config.update(config)
    cherrypy.config.update({'server.socket_port': serverPort})
    cherrypy.quickstart(MyWebService(), '/', conf)



'''
        '/static': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': 'public'
        },
        '/css': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': "public/css"
        },
        '/js': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': "public/js"
        },
'''
