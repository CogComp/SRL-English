
ID_OUTPUT="$1-id-output.txt"
SRL_INPUT="$1-srl-input.txt"

allennlp predict test-id-bert/model.tar.gz $1 --output-file ${ID_OUTPUT} --cuda-device 0 --predictor "nombank-id" --include-package id_nominal
python convert_id_to_srl_input.py ${ID_OUTPUT} ${SRL_INPUT}
if [ -z "$2" ]
then
	allennlp predict nom-srl-bert/model.tar.gz ${SRL_INPUT} --cuda-device 0 --predictor "nombank-semantic-role-labeling" --include-package nominal_srl
else 
	allennlp predict nom-srl-bert/model.tar.gz ${SRL_INPUT} --output-file $2 --cuda-device 0 --predictor "nombank-semantic-role-labeling" --include-package nominal_srl

fi

