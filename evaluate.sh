output="runs"
device="cpu"

# django dataset
echo "evaluating django decode results"
datatype="django"
dataset="data/django.cleaned.dataset.freq5.par_info.refact.space_only.bin"
model="model.npz"

# evaluate the decoding result
python code_gen.py \
	-data_type ${datatype} \
	-data ${dataset} \
	-output_dir ${output} \
	evaluate \
	-input ${output}/${model}.decode_results.test.bin | tee ${output}/${model}.decode_results.test.log
