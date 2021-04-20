This repository contains all files created to perform the BERT-based nominal SRL, both using the Nombank dataset and the Ontonotes dataset. It also includes a BERT-based predicate identifier based on the Nombank dataset.

[A visualized demo of this code is running on the CogComp demo site](https://cogcomp.seas.upenn.edu/page/demo_view/EnglishSRL).

For Nombank: It includes files to read the `nombank.1.0` corpus into a format usable by the model, as well as a reader, model, and predictor to be used with the AllenNLP workflow.
For Ontonotes: It includes the files to read the CoNLL-formatted Ontonotes, model, and predictor to be used with the AllenNLP workflow. 

# Set up the environment
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

# Pre-process the Nombank data
You must first obtain the Nombank data from the [site](https://nlp.cs.nyu.edu/meyers/NomBank.html).

Downloads needed: nltk

Set up the data and run preprocessing. (This will take about (25?) minutes.): 
First sort the nombank.1.0 file: `sort nombank.1.0 > sorted_nombank.1.0`
Then enter the `preprocess_nombank/preprocess_nombank.py` file and update the path pointer links to where you put the data. Once this has been set up, run preprocessing:
```
cd preprocess_nombank
python preprocess_nombank.py
```

# Obtain the CONLL data
[Follow these instructions.](https://cemantix.org/data/ontonotes.html)

# Run the model: Nombank

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
allennlp predict nom-srl-bert/model.tar.gz input.txt --output-file predicted_output.txt --predictor "nominal-semantic-role-labeling" --include-package nominal_srl
```

To predict a single sentence without a file input. With the -ta flag, output will be in TextAnnotation form. Otherwise, it will be in the standard output dictionary form:
```
python nom_predict_sentence.py nom-srl-bert/model.tar.gz -s "By this September, program traders were doing a record 13.8% of the Big Board's average daily trading volume." -i 5 11 -o predict_nom_output.txt
```

## Run the model: Nombank Sense Disambiguation and SRL

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
# Run the model: Ontonotes

First set up the paths used. The existing `set_paths.sh` file does work on the CCG machines for the location of the unpacked CONLL data, as of 7/27/2020. If you would like to point to a different location, simply modify the `set_path.sh` file.

## The following instructions are to replicate the AllenNLP implementation of verb srl

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
## To run the Ontonotes model on just the NN predicates, you can do the following:


To predict a single sentence without file input:
```
python onto_nom_predict_sentence.py srl-bert-ontonotes/model.tar.gz "Ideal Basic Industries Inc. said its directors reached an agreement in principle calling for HOFI North America Inc. to combine its North American cement holdings with Ideal in a transaction that will leave Ideal's minority shareholders with 12.8% of the combined company." --output_file onto_predict_nom_output.txt
```
## Run the model: Ontonotes Sense Disambiguation and SRL

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

# Run the nominal identifier model:

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

# Run the entire nominal identifier and srl pipeline:


To run the nominal predicate identifier and srl predictor together, a simple script and function is provided to stitch the output of the id model (at `/shared/celine/test_allennlp/v0.9.0/test-id-bert/model.tar.gz` as of 8/13/20) to the input of the srl model (at `/shared/celinel/test_allennlp/v0.9.0/nom-srl-bert/model.tar.gz` as of 8/13/20).

Make a file of the input sentences in the same dict format: `{"sentence": "This part will be the sentence."}`. In the example below, we name the input file with this format as `input.txt`. 
```
. ./run_nom_pipeline.sh input.txt
```
The output will be printed to the terminal. Or, if you would like for the output to be printed to a file, you can specify the location with the following command:
```
. ./run_nom_pipeline.sh input.txt output.txt
```

# Run the cherrypy backend:
The cherrypy backend runs the predictors for nominal identification, nominal sense SRL, verb sense SRL, and preposition SRL. 

Set up the environment and set up the port. Make modifications to the port number inside `backend.py` as necessary.
```
pip install cherrypy
python backend.py
```
Then in another terminal window, run the program with either the following, modifying the port number and sentence as necessary. The following curl commands are supported:
```
curl -d '{"sentence": "The president of the USA holds a lot of power."}' -H "Content-Type: application/json" -X POST http://localhost:8043/generate_table
curl -d 'The president of the USA holds a lot of power.' -H "Content-Type: text/plain" -X GET http://localhost:8043/annotate
curl -d '{"sentence": "The president of the USA holds a lot of power."}' -H "Content-Type: application/json" -X POST http://localhost:8043/annotate
curl -X GET http://localhost:8043/annotate?sentence=The%20president%20of%20the%20USA%20holds%20a%20lot%20of%20power.

```

Alternatively, to run it from a browser, navigate to `http://localhost:8043` and see the input/output on the screen.
