export CUDA_VISIBLE_DEVICES=1
while true
do
	python backend.py 0.0.0.0 4039
	# python backend_gpu.py 0.0.0.0 4039
done
