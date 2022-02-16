export CUDA_VISIBLE_DEVICES=1,2
while true
do
	# python backend_gpu.py --gpu "1" --url "0.0.0.0" --port "4044"
	python backend_gpu.py
	# sleep 2
done
