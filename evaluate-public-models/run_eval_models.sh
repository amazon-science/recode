# model_names="codegen-6B-multi"
model_names="codegen-350M-multi"
# model_names="incoder-1B"
#model_names="incoder-6B"
#model_names="gpt-j-6B"
datasets="humaneval" # or mbpp or new or perturbed dataset names

num_samples=0 # zero means greedy
top_p=0.95
temperature=0.2

test_file=$1
output_dir=$2
datasets=$3
model_names=$4
num_gpu="${5:-1}"
override="${6:-False}"
# echo ngpu: $num_gpu
# echo override: $override

for dataset in $datasets; do
    if [ $num_samples == 0 ]; then
        policy=greedy
        sample_args="--num_samples_per_example 1"
    else
        policy=sampling
        sample_args="--top_p $top_p --temperature $temperature --do_sample --num_samples_per_example $num_samples --do_rank"
    fi

    # add new or perturbed datasets here
    if [ $dataset == humaneval ]; then
        # test_file=datasets/nominal/HumanEval.jsonl
        lang=python
    elif [ $dataset == mbpp ]; then
        # test_file=datasets/nominal/mbpp_wtest.jsonl
        lang=python
    else
        echo "unkown dataset $dataset"
        exit
    fi
    output_dir=${output_dir}/${policy}

    for model_name in $model_names; do
        if [[ "$model_name" == *"codegen"* ]]; then
            model_provider=Salesforce
        elif [[ "$model_name" == *"incoder"* ]]; then
                model_provider=facebook
        elif [[ "$model_name" == *"gpt-j"* ]]; then
                model_provider=EleutherAI
            else
                echo "unkown model $model_name"
                exit
            fi

        if [[ "$model_name" != *"incoder"* ]]; then
            sample_args="$sample_args --use_stopping_criteria"
        fi

        if [ $override == True ]; then
            sample_args="$sample_args --override_previous_results"
        fi
        echo sample_args: $sample_args
        # exit
	
	#if false; then
	# generate
        python3 -m torch.distributed.run --nproc_per_node $num_gpu --master_port 20000 evaluate-public-models/evaluate_model.py \
		--model_name_or_path $model_provider/$model_name \
		--tokenizer_name $model_provider/$model_name \
		--output_dir $output_dir \
		--programming_lang $lang \
		--test_file $test_file \
		--fp16 \
		$sample_args
	#fi

        # evaluate
        python3 evaluate-public-models/evaluate_model.py \
		--model_name_or_path $model_provider/$model_name \
		--tokenizer_name $model_provider/$model_name \
		--output_dir $output_dir \
		--programming_lang $lang \
		--test_file $test_file \
		--run_eval_only
    done
done
