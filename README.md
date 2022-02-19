This repository contains all files created to perform the BERT-based nominal SRL, both using the Nombank dataset and the Ontonotes dataset. It also includes a BERT-based predicate identifier based on the Nombank, STREUSLE, and Bolt datasets. The design of the models in this repository are based on a BERT + linear layer model used in ['Simple BERT Models for Relation Extraction and Semantic Role Labeling'](https://arxiv.org/pdf/1904.05255.pdf).

[A visualized demo of this code is running on the CogComp demo site](https://cogcomp.seas.upenn.edu/page/demo_view/EnglishSRL).

[Trained models can be downloaded from Huggingface] (https://huggingface.co/Yuqian/Celine_SRL/tree/main)
Trained models can also be downloaded from the CogComp website:
[Nominal Sense Disambiguation and SRL](https://cogcomp.seas.upenn.edu/models/English_SRL_Sense/nom-sense-srl/model.tar.gz)
[Verb Sense Disambiguation and SRL](https://cogcomp.seas.upenn.edu/models/English_SRL_Sense/verb-sense-srl/model.tar.gz)
<!-- [](https://cogcomp.seas.upenn.edu/models/English_SRL_Sense/test-id-bert/model.tar.gz) -->
[Preposition SRL](https://cogcomp.seas.upenn.edu/models/English_SRL_Sense/prep-srl/model.tar.gz)

For Nombank: It includes files to read the `nombank.1.0` corpus into a format usable by the model, as well as a reader, model, and predictor to be used with the AllenNLP workflow. 
For Ontonotes: It includes the files to read the CoNLL-formatted Ontonotes, model, and predictor to be used with the AllenNLP workflow. 


## Setup Environment
Create the environment and set up installs. I do this with a python3 venv.
```
python3 -m venv srl_venv
cd srl_venv
source bin/activate
pip install allennlp==0.9.0
```

The GPUs on the CCG machines are CUDA version 10.1, so we set Pytorch back to version 1.4:
```
pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install nltk # Only need this if you are running the data pre-processing step.
git clone <this repository's link>
cd nominal-srl-allennlpv0.9.0
```
## Data
### Pre-process Nombank Data
You must first obtain the Nombank data from the [site](https://nlp.cs.nyu.edu/meyers/NomBank.html).

Downloads needed: nltk

Set up the data and run preprocessing. (This will take about (25?) minutes.): 
First sort the nombank.1.0 file: `sort nombank.1.0 > sorted_nombank.1.0`
Then enter the `preprocess_nombank/preprocess_nombank.py` file and update the path pointer links to where you put the data. Once this has been set up, run preprocessing:
```
cd preprocess_nombank
python preprocess_nombank.py
```

### Obtain the CONLL data
[Follow these instructions.](https://cemantix.org/data/ontonotes.html)


## The folders and views
The view `SRL_VERB` corresponds to the verbal sense SRL. The model is created in folder `verb_sense_srl`. Without sense information, the folder to use would be the one provided in [AllenNLP v0.9.0 SRL](https://github.com/allenai/allennlp/blob/v0.9.0/allennlp/models/srl_bert.py).

The view `SRL_NOM` corresponds to the nominal sense SRL. We pass the sentence through a nominal identifier model created in `id_nominal`, then perform SRL on the identified nominals. The model is created in folder `nominal_sense_srl`, with predictor `predictor.py`. Without sense information, the folder to use would be `nominal_srl`.

The view `SRL_NOM_ALL` corresponds to the nominal sense SRL where we perform SRL on all identified nouns in the sentences. The model is created in folder `nominal_sense_srl`, with predictor `predictor_all.py`. Without sense information, the folder to use would be `nominal_srl`.

The view `SRL_PREP` corresponds to the preposition SRL. The model is created in folder `prep_srl`.

INCOMPLETE: `bolt_srl` is the beginning of an effort to perform SRL on adjectives. `bolt.py` helps read in the Bolt data.

NOT INCLUDED: `combined_unconstrained_srl` is an effort to train an SRL model on all data we have: Ontonotes (verbs + some nominals), Nombank (nominals), Bolt (verbs + nominals + adjectives), Streusle (prepositions). This model has been used to perform some analysis, but it has not been fully documented so is currently considered incomplete. 

## Models
The design of the models in this repository are based on a BERT + linear layer model used in ['Simple BERT Models for Relation Extraction and Semantic Role Labeling'](https://arxiv.org/pdf/1904.05255.pdf). We take a string input, tokenize it, pass the tokenized input through the BERT module, pass the output of the BERT module through a linear layer, then use an Adam loss to backpropagate. 

In the joint SRL + sense models, we also use this BERT + linear design. The SRL and sense predictions share the BERT layer then each have their own separate linear layers. The Adam losses of the SRL prediction and the sense prediction are added to create a joint loss for backpropagation.

### Run model: Nombank

Set up the paths and model files, which are at `nom_bert_srl.jsonnet`. Set the `train_data_path` and `validation_data_path` to the absolute paths to the files generated in the previous preprocessing step. 

Navigate back to the outer `nominal-srl-allennlpv0.9.0` folder. 

Train the model:
```
. ./set_path.sh # Feel free to update any of the paths to your needs.
allennlp train nom_bert_srl.jsonnet -s nom-srl-bert-test -f --include-package nominal_srl
```
(An already-trained version of the model achieving 0.835 F1 on the Nombank test dataset can be found at `/shared/celinel/test_allennlp/v0.9.0/nom-srl-bert/model.tar.gz` as of 7/27/20.)

To evaluate:
```
allennlp evaluate nom-srl-bert/model.tar.gz /path/to/preprocess_nombank/test.srl --output-file nom-srl-bert/evaluation.txt --include-package nominal_srl
```

To predict:
Create input text file with JSON formatting: `{"sentence": "This is a sentence.", "indices": [3]}` for each sentence you would like predicted.
```
allennlp predict nom-srl-bert/model.tar.gz input.txt --output-file predicted_output.txt --predictor "nombank-semantic-role-labeling" --include-package nominal_srl
```

To predict a single sentence without a file input. With the -ta flag, output will be in TextAnnotation form. Otherwise, it will be in the standard output dictionary form:
```
python nom_predict_sentence.py nom-srl-bert/model.tar.gz -s "By this September, program traders were doing a record 13.8% of the Big Board's average daily trading volume." -i 5 11 -o predict_nom_output.txt
```

#### Run model: Nombank Sense Disambiguation and SRL

There is also a model that joinly performs sense disambiguation and SRL, defined at `nominal_sense_srl`. Set up the paths and model files at `nom_sense_srl.jsonnet`.  

Train the model:
```
allennlp train nom_sense_srl.jsonnet -s nom-sense-srl -f --include-package nominal_sense_srl
```
(An already-trained version of the model achieving 0.824 F1 on SRL and 0.979 accuracy on sense on the Nombank test dataset can be found at `/shared/celinel/test_allennlp/v0.9.0/nom-sense-srl/model.tar.gz` as of 8/21/20.)

To evaluate:
```
allennlp evaluate nom-sense-srl/model.tar.gz /path/to/preprocess_nombank/test.srl --output-file nom-sense-srl/evaluation.txt --include-package nominal_sense_srl
```

To predict:
Create input text file with JSON formatting: `{"sentence": "This is a sentence.", "indices": [3]}` for each sentence you would like predicted.
```
allennlp predict nom-sense-srl/model.tar.gz input.txt --output-file nom-sense-srl/predicted_output.txt --predictor "nominal-sense-srl" --include-package nominal_sense_srl
```
### Run model: Ontonotes

First set up the paths used. The existing `set_paths.sh` file does work on the CCG machines for the location of the unpacked CONLL data, as of 7/27/2020. If you would like to point to a different location, simply modify the `set_path.sh` file.

#### Instructions to Replicate AllenNLP Implementation of Verb SRL

To train: 
```
. ./set_paths.sh
allennlp train bert_base_srl.jsonnet -s srl-bert-ontonotes -f
```
(An already-trained version of the model achieving 0.862 F1 can be found at `/shared/celinel/test_allennlp/v0.9.0/verb-srl-bert/model.tar.gz` as of 7/27/2020.)

To evaluate:
```
allennlp evaluate srl-bert-ontonotes/model.tar.gz /path/to/conll/formatted/ontonotes/test --output-file srl-bert-ontonotes/evaluation.txt
```

To predict with JSON-formatted input file:
```
allennlp predict srl-bert-ontonotes/model.tar.gz input.txt --output-file onto_predicted_output.txt
```

To predict a single sentence without file input:
```
python onto_predict_sentence.py srl-bert-ontonotes/model.tar.gz "Ideal Basic Industries Inc. said its directors reached an agreement in principle calling for HOFI North America Inc. to combine its North American cement holdings with Ideal in a transaction that will leave Ideal's minority shareholders with 12.8% of the combined company." --output_file onto_predict_output.txt
```
#### To Run Ontonotes Model on only NN Predicates:


To predict a single sentence without file input:
```
python onto_nom_predict_sentence.py srl-bert-ontonotes/model.tar.gz "Ideal Basic Industries Inc. said its directors reached an agreement in principle calling for HOFI North America Inc. to combine its North American cement holdings with Ideal in a transaction that will leave Ideal's minority shareholders with 12.8% of the combined company." --output_file onto_predict_nom_output.txt
```
#### Run Model: Ontonotes Sense Disambiguation and SRL

There is also a model that joinly performs sense disambiguation and SRL, defined at `verb_sense_srl`. Set up the paths and model files at `bert_sense_srl.jsonnet`.  

To train: 
```
. ./set_paths.sh
allennlp train bert_sense_srl.jsonnet -s verb-sense-srl -f --include-package verb_sense_srl
```
(An already-trained version of the model achieving 0.84 F1 on SRL and 0.88 accuracy on sense can be found at `/shared/celinel/test_allennlp/v0.9.0/verb-sense-srl/model.tar.gz` as of 8/21/2020.)

To evaluate:
```
allennlp evaluate verb-sense-srl/model.tar.gz /path/to/conll/formatted/ontonotes/test --output-file verb-sense-srl/evaluation.txt --include-package verb_sense_srl
```

To predict with JSON-formatted input file:
```
allennlp predict verb-sense-srl/model.tar.gz input.txt --output-file verb-sense-srl/onto_predicted_output.txt --predictor "sense-semantic-role-labeling" --include-package verb_sense_srl
```

### Run Model: Nominal Identifier

Set up the paths and model files, which are at `bert_nom_id.jsonnet`. Set the `train_data_path`, `validation_data_path`, and `test_data_path` to the absolute paths to the files generated in the Nombank pre-processing step.

Train the model:
```
allennlp train bert_nom_id.jsonnet -s nom-id-bert -f --include-package id_nominal
``` 
(An already-trained version of the model achieving 0.81 F1 on Nombank can be found at `/shared/celinel/test_allennlp/v0.9.0/test-id-bert/model.tar.gz` as of 8/13/20.)

To evaluate:
```
allennlp evaluate nom-id-bert/model.tar.gz /path/to/preprocessed_nombank/test.srl --output-file nom-id-bert/evaluation.txt --include-package nominal_id
```

To predict: Create input text file with JSON formatting: `{"sentence": "This is a sentence."}` for each sentence that you would like predicted.
```
allennlp predict nom-id-bert/model.tar.gz input.txt --output-file predicted_output.txt --predictor "nombank-id" --include-package nominal_id
```

### Run Entire Nominal Identifier and SRL Pipeline:


To run the nominal predicate identifier and srl predictor together, a simple script and function is provided to stitch the output of the id model (at `/shared/celine/test_allennlp/v0.9.0/test-id-bert/model.tar.gz` as of 8/13/20) to the input of the srl model (at `/shared/celinel/test_allennlp/v0.9.0/nom-srl-bert/model.tar.gz` as of 8/13/20).

Make a file of the input sentences in the same dict format: `{"sentence": "This part will be the sentence."}`. In the example below, we name the input file with this format as `input.txt`. 
```
. ./run_nom_pipeline.sh input.txt
```
The output will be printed to the terminal. Or, if you would like for the output to be printed to a file, you can specify the location with the following command:
```
. ./run_nom_pipeline.sh input.txt output.txt
```

## Run cherrypy backend:
The cherrypy backend runs the predictors for nominal identification, nominal sense SRL, verb sense SRL, and preposition SRL. 

Set up the environment and set up the port. Make modifications to the port number inside `backend.py` as necessary.
```
pip install cherrypy
python backend.py
```
Then in another terminal window, run the program with either the following, modifying the port number and sentence as necessary. The following curl commands are supported:
```
curl -d 'The president of the USA holds a lot of power.' -H "Content-Type: text/plain" -X GET http://localhost:8043/annotate
curl -d '{"sentence": "The president of the USA holds a lot of power."}' -H "Content-Type: application/json" -X POST http://localhost:8043/annotate
curl -X GET http://localhost:8043/annotate?sentence=The%20president%20of%20the%20USA%20holds%20a%20lot%20of%20power.
```

Alternatively, to run it from a browser, navigate to `http://localhost:8043` and see the input/output on the screen.


## Performance Metrics of SRL

| Model Name                |Dataset      | Precision(%) | Recall(%)   | F<sub>1</sub> Score(%) | Loss        | Sense Accuracy(%) | Combined Score(%)|
|:---------------------:|:-----------:|:------------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| Combined Unconstrained| ---         | 83.936       | 85.499      | 84.710      | 0.3552      | ---         | ---         |
| ID Nominal            | ---         | ---          | ---         | ---         | ---         | ---         | ---         |
| Nominal Sense         | Nombank     | 80.688       | 83.664      | 82.149      | 0.4100      | 97.900      | 80.424      |
| Nominal               | Nombank     | 81.139       | 83.428      | 82.268      | 0.3218      | ---         | ---         |
| Ontonotes             | ---         | ---          | ---         | ---         | ---         | ---         | ---         |
| Prep                  | ---         | ---          | ---         | ---         | ---         | ---         | ---         |
| Verb Sense            | CONLL       |82.946        | 85.381      | 84.146      | 1.0142      | 88.142      | 74.168      |


# Contact:
Questions: contact Celine at celine.y.lee@gmail.com
