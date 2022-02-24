from allennlp.predictors.predictor import JsonDict
from allennlp.common.util import sanitize
#from nltk.stem import WordNetLemmatizer
import spacy
import numpy

import xml.etree.ElementTree as ET
import predict_utils
import os

class TabularView(object):
    def __init__(self):
        super().__init__()
        
        self.ta = {"corpusId": "", "id": ""}
        self.views = {}
        self.current_outputs = {}
        self.sp =spacy.load('en_core_web_sm')
    
    def update_sentence(self, nom_srl_output):
        generator = "srl_pipeline"
        tokens = nom_srl_output["words"]
        text = " ".join(tokens)
        self.ta["text"] = text
        self.ta["tokens"] = tokens
        self.ta["tokenOffsets"] = predict_utils.create_token_char_offsets(text)
        #sentence_end_positions = [i+1 for i,x in enumerate(tokens) if x=="."]
        sentence_end_positions = [i+1 for i,x in enumerate(tokens) if x in [".", "?", "!", "..."]]
        sentences = {"generator": generator, "score": 1.0, "sentenceEndPositions": sentence_end_positions}
        self.ta["sentences"] = sentences
        
        self.views = {}
        self.views["SENTENCE"] = predict_utils.create_sentence_view(tokens)
        self.views["TOKENS"] = predict_utils.create_tokens_view(tokens)
        self.ta["views"] = self.views.values()

    def update_view(self, view_name, srl_output):
        if view_name.startswith("SRL_NOM"):
            output = srl_output["nominals"]
            self.views[view_name] = self._create_srl_nom_view(output, view_name)
        elif view_name == "SRL_VERB":
            output = srl_output["verbs"]
            self.views[view_name] = self._create_srl_verb_view(output, srl_output["words"])
        elif view_name == "SRL_PREP":
            output = srl_output["prepositions"]
            self.views[view_name] = self._create_srl_prep_view(output)
        self.current_outputs[view_name] = output
        self.ta["views"] = list(self.views.values())

    def remove_view(self, view_name):
        if view_name in self.views:
            del self.views[view_name]
        if view_name in self.current_outputs:
            del self.current_outputs[view_name]

    def clear_table(self):
        self.views = {}
        self.ta = {"corpusId": "", "id": "", "text": "", "tokens": [], "tokenOffsets": [], "sentences": {}, "views": []}
        self.current_outputs = {}


    def get_textannotation(self):
        sanitized = self._sanitize(self.ta)
        # print(sanitized)
        # print(type(sanitized))
        return sanitized

    
    def _sanitize(self,x):
        if isinstance(x, (str, float, int, bool)):
            return x
        elif isinstance(x, numpy.ndarray):
            return x.tolist()
        elif isinstance(x, numpy.number):
            return x.item()
        elif isinstance(x, dict):
            return {key:self._sanitize(value) for key, value in x.items()}
        elif isinstance(x, numpy.bool_):
            return bool(x)
        elif isinstance(x, (list, tuple)):
            return [self._sanitize(x_i) for x_i in x]
        elif x is None:
            return "None"
        elif hasattr(x, "to_json"):
            return x.to_json()
        else:
            print(x, ' IS THE HARD ONE WE CANOT SANITIZE, IT IS OF TYPE, ', type(x))
    
    def _get_sense_description(self, directory, predicate, sense):
        subsense = 0
        if sense.find('.') >= 0:
            sense = sense[:sense.find('.')]
            subsense = int(sense[sense.find('.')+1:])
        if sense.isdigit():
            sense = int(sense)
        sense_name = "NA"
        sense_descriptions = {}
        if directory == "NOMBANK":
            frame_file = "/shared/celinel/noun_srl/nombank.1.0/frames/"
            # predicate = self.sp(predicate)[0].lemma_
        elif directory == "PROPBANK":
            frame_file = "/shared/celinel/propbank-frames/frames/"
        elif directory == "ONTONOTES":
            frame_file = "/shared/celinel/LDC2013T19/ontonotes-release-5.0/data/files/data/english/metadata/frames/"
            predicate = predicate + "-v"
        else:
            return sense_name, sense_descriptions
        frame_file = frame_file + predicate + ".xml"
        if os.path.isfile(frame_file):
            tree = ET.parse(frame_file)
            root = tree.getroot()
            for roleset in root.findall('predicate/roleset'):
                roleset_id = roleset.get('id')
                found_sense = roleset_id[roleset_id.find('.')+1:]
                found_subsense = 0
                if found_sense.find('.') >= 0:
                    found_sense = found_sense[:found_sense.find('.')]
                    found_subsense = int(found_sense[found_sense.find('.')+1:])
                    # TODO subsense
                if found_sense.isdigit():
                    found_sense = int(found_sense)
                # print('FOUND SENSE: ', found_sense, ' LOOKING FOR ', sense)
                if found_sense != sense:
                    continue
                if (sense == "LV" and found_sense == "LV") or (int(float(sense)) == int(float(found_sense))):
                    sense_name = roleset.get('name')
                    for role in roleset.findall('roles/role'):
                        sense_descriptions[role.attrib['n']] = role.attrib['descr']
                    break
                else:
                    continue
        return sense_name, sense_descriptions

    def _get_frame_descriptions(self):
        frames = []
        for view_name, output in self.current_outputs.items():
            no_frame_files = False
            if view_name.startswith("SRL_NOM"):
                frame_file = "/shared/celinel/noun_srl/nombank.1.0/frames/"
            elif view_name == "SRL_VERB":
                frame_file = "/shared/celinel/LDC2013T19/ontonotes-release-5.0/data/files/data/english/metadata/frames/"
            elif view_name == "SRL_PREP":
                no_frame_files = True
            else:
                return 
            for frame in output:
                if not frame:
                    continue
                frame_info = {}
                if "sense" not in frame:
                    frames.append(frame_info)
                    continue
                if "nominal" in frame:
                    predicate = frame["nominal"]
                    sentence = self.sp(predicate)
                    predicate = sentence[0].lemma_
                elif "verb" in frame:
                    predicate = frame["verb"]
                    sentence = self.sp(predicate)
                    predicate = sentence[0].lemma_ + "-v"
                elif "preposition" in frame:
                    predicate = frame["preposition"]
                    sentence = self.sp(predicate)
                    predicate = sentence[0].lemma_
                    frames.append(frame_info)
                    continue

                current_frame_file = frame_file + predicate + ".xml"
                # frame_info = ""
                if os.path.isfile(current_frame_file):
                    tree = ET.parse(current_frame_file)
                    root = tree.getroot()
                    for roleset in root.findall('predicate/roleset'):
                        roleset_id = roleset.get('id')
                        sense = roleset_id[roleset_id.find('.')+1:]
                        subsense = 0
                        if sense.find('.') >= 0:
                            sense = sense[:sense.find('.')]
                            subsense = int(sense[sense.find('.')+1:])
                            # TODO subsense
                        frame_sense = frame["sense"]
                        frame_subsense = 0
                        if frame_sense.find('.') >= 0:
                            frame_sense = frame_sense[:frame_sense.find('.')]
                            frame_subsense = int(frame_sense[frame_sense.find('.')+1:])
                        if frame_sense == "LV" or sense == "LV":
                            continue
                        if (sense == "LV" and frame_sense == "LV") or (int(float(sense)) == int(float(frame_sense))):
                            frame_info["name"] = roleset.get('name')
                            for role in roleset.findall('roles/role'):
                                frame_info[role.attrib['n']] = role.attrib['descr']
                            break
                        else:
                            continue
                frames.append(frame_info)
                # html_string += "<td>" + frame_info + "</td>"
        return frames


    def _create_srl_nom_view(self, nom_srl_frames, view_name):
        srl_nom_view = {"viewName": view_name}
        constituents = []
        relations = []
        for frame in nom_srl_frames:
            predicate = frame["nominal"]
            predicate = self.sp(predicate)[0].lemma_
            # print(frame["nominal"], '->', predicate)
            description = frame["description"]
            tags = frame["tags"]
            sense = "NA"
            if "sense" in frame:
                sense = str(frame["sense"])
            predicate_idx = frame["predicate_index"]
            sense_name, sense_descriptions = self._get_sense_description("NOMBANK", predicate, sense)
            properties = {"SenseNumber": sense, "predicate": predicate, "sense": sense_name}
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
                        sense_description = ""
                        if active_tag.startswith("ARG") and active_tag[3].isdigit():
                            if active_tag[3] in sense_descriptions:
                                sense_description = sense_descriptions[active_tag[3]]
                            else:
                                sense_description = "NA"
                        elif "M-" in active_tag:
                            sense_description = "Modifier"
                        constituent = {"label": active_tag, "score": 1.0, "start": active_tag_start_idx, "end": tag_idx}
                        relation = {"relationName": active_tag, "srcConstituent": predicate_constituent_idx, "targetConstituent": len(constituents), "sense": sense_description}
                        relations.append(relation)
                        constituents.append(constituent)
                        active_tag = ""
                        active_tag_start_idx = -1
                    continue
                if tag[2:] == active_tag:
                    continue
                else:
                    if active_tag != "":
                        sense_description = ""
                        if active_tag.startswith("ARG") and active_tag[3].isdigit():
                            if active_tag[3] in sense_descriptions:   
                                sense_description = sense_descriptions[active_tag[3]]  
                            else:
                                sense_description = "NA"
                        elif "M-" in active_tag:   
                            sense_description = "Modifier"
                        constituent = {"label": active_tag, "score": 1.0, "start": active_tag_start_idx, "end": tag_idx}
                        relation = {"relationName": active_tag, "srcConstituent": predicate_constituent_idx, "targetConstituent": len(constituents), "sense": sense_description}
                        relations.append(relation)
                        constituents.append(constituent)
                    active_tag = tag[2:]
                    active_tag_start_idx = tag_idx
            # collect stragglers
            if active_tag != "":
                sense_description = ""
                if active_tag.startswith("ARG") and active_tag[3].isdigit():  
                    if active_tag[3] in sense_descriptions:    
                        sense_description = sense_descriptions[active_tag[3]]    
                    else:
                        sense_description = "NA"
                elif "M-" in active_tag:
                    sense_description = "Modifier"
                constituent = {"label": active_tag, "score": 1.0, "start": active_tag_start_idx, "end": len(tags)}
                relation = {"relationName": active_tag, "srcConstituent": predicate_constituent_idx, "targetConstituent": len(constituents), "sense": sense_description}
                relations.append(relation)
                constituents.append(constituent)
        nom_view_data = [{"viewType": "", "viewName": view_name, "generator": "nominal_srl_pipeline", "score": 1.0, "constituents": constituents, "relations": relations}]
        srl_nom_view["viewData"] = nom_view_data
        return srl_nom_view
    
    def _create_srl_verb_view(self, verb_srl_frames, sentence):
        srl_verb_view = {"viewName": "SRL_ONTONOTES"}
        constituents = []
        relations = []
        for frame in verb_srl_frames:
            verb = frame["verb"]
            lemma = self.sp(verb)[0].lemma_
            # print(verb, '->', lemma)
            description = frame["description"]
            tags = frame["tags"]
            sense = "NA"
            if "sense" in frame:
                sense = str(frame["sense"])
            sense_name, sense_descriptions = self._get_sense_description("ONTONOTES", lemma, sense)
            properties = {"SenseNumber": sense, "predicate": lemma, "sense": sense_name}
            if "B-V" not in tags: # NOTE This might give the wrong index of verb.
                predicate_indices = [i for i, elt in enumerate(sentence) if elt == verb]
                for pred_idx in predicate_indices:
                    if tags[pred_idx] != "O":
                        predicate_idx = pred_idx
                        continue
            else:
                predicate_idx = tags.index("B-V")
            
            constituent = {"label": "Predicate", "score": 1.0, "start": predicate_idx, "end": predicate_idx+1, "properties": properties}
            predicate_constituent_idx = len(constituents)
            constituents.append(constituent)
            active_tag = ""
            active_tag_start_idx = -1
            for tag_idx, tag in enumerate(tags):
                if tag in {"O", "B-V"}:
                    if active_tag != "":
                        sense_description = ""
                        if active_tag.startswith("ARG") and active_tag[3].isdigit():
                            if active_tag[3] in sense_descriptions:
                                sense_description = sense_descriptions[active_tag[3]]
                            else:
                                sense_description = "NA"
                        elif "M-" in active_tag:
                            sense_description = "Modifier"
                        constituent = {"label": active_tag, "score": 1.0, "start": active_tag_start_idx, "end": tag_idx}
                        relation = {"relationName": active_tag, "srcConstituent": predicate_constituent_idx, "targetConstituent": len(constituents), "sense": sense_description}
                        relations.append(relation)
                        constituents.append(constituent)
                        active_tag = ""
                        active_tag_start_idx = -1
                    continue
                if tag[2:] == active_tag:
                    continue
                else:
                    if active_tag != "":
                        sense_description = ""
                        if active_tag.startswith("ARG") and active_tag[3].isdigit():
                            if active_tag[3] in sense_descriptions:
                                sense_description = sense_descriptions[active_tag[3]]
                            else:
                                sense_description = "NA"
                        elif "M-" in active_tag:
                            sense_description = "Modifier"
                        constituent = {"label": active_tag, "score": 1.0, "start": active_tag_start_idx, "end": tag_idx}
                        relation = {"relationName": active_tag, "srcConstituent": predicate_constituent_idx, "targetConstituent": len(constituents), "sense": sense_description}
                        relations.append(relation)
                        constituents.append(constituent)
                    active_tag = tag[2:]
                    active_tag_start_idx = tag_idx
            # collect stragglers
            if active_tag != "":
                sense_description = ""
                if active_tag.startswith("ARG") and active_tag[3].isdigit():
                    if active_tag[3] in sense_descriptions:
                        sense_description = sense_descriptions[active_tag[3]]
                    else:
                        sense_description = "NA"
                elif "M-" in active_tag:
                    sense_description = "Modifier"
                constituent = {"label": active_tag, "score": 1.0, "start": active_tag_start_idx, "end": len(tags)}
                relation = {"relationName": active_tag, "srcConstituent": predicate_constituent_idx, "targetConstituent": len(constituents), "sense": sense_description}
                relations.append(relation)
                constituents.append(constituent)
        verb_view_data = [{"viewType": "", "viewName": "SRL_ONTONOTES", "generator": "srl_pipeline", "score": 1.0, "constituents": constituents, "relations": relations}]
        srl_verb_view["viewData"] = verb_view_data
        return srl_verb_view


    def _create_srl_prep_view(self, prep_srl_frames):
        srl_prep_view = {"viewName": "PREPOSITION_SRL"}
        constituents = []
        relations = []
        for frame in prep_srl_frames:
            predicate = frame["preposition"]
            desciption = frame["description"]
            tags = frame["tags"]
            sense = "NA"
            if "sense" in frame:
                sense = frame["sense"]
            predicate_idx = frame["predicate_index"]
            properties = {"SenseNumber": sense, "predicate": predicate, "sense": "NA"}
            if len(predicate_idx) > 1:
                print("Multiple indices of predicate. Using first.")
            constituent = {"label": "Predicate", "score": 1.0, "start": predicate_idx[0], "end": predicate_idx[0] + 1, "properties": properties}
            predicate_constituent_idx = len(constituents)
            constituents.append(constituent)
            active_tag = ""
            active_tag_start_idx = -1
            for tag_idx, tag in enumerate(tags):
                if tag in {"O", "B-PREP"}:
                    if active_tag != "":
                        constituent = {"label": active_tag, "score": 1.0, "start": active_tag_start_idx, "end": tag_idx}
                        relation = {"relationName": active_tag, "srcConstituent": predicate_constituent_idx, "targetConstituent": len(constituents), "sense": "NA"}
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
                        relation = {"relationName": active_tag, "srcConstituent": predicate_constituent_idx, "targetConstituent": len(constituents), "sense": "NA"}
                        relations.append(relation)
                        constituents.append(constituent)
                    active_tag = tag[2:]
                    active_tag_start_idx = tag_idx
            # collect stragglers
            if active_tag != "":
                constituent = {"label": active_tag, "score": 1.0, "start": active_tag_start_idx, "end": len(tags)}
                relation = {"relationName": active_tag, "srcConstituent": predicate_constituent_idx, "targetConstituent": len(constituents), "sense": "NA"}
                relations.append(relation)
                constituents.append(constituent)
        prep_view_data = [{"viewType": "", "viewName": "PREPOSITION_SRL", "generator": "preposition_srl_pipeline", "score": 1.0, "constituents": constituents, "relations": relations}]
        srl_prep_view["viewData"] = prep_view_data
        return srl_prep_view
