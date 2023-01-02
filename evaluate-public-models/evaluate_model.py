import argparse
import json
import os
import torch
from tqdm import tqdm
from utils.metrics import read_problems, run_passatk_eval
from utils.truncate import filter_valid_code, inference_cut_off
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.distributed as dist
# from stopping_criterion.stopping_criterion import get_stopping_criteria_per_language, SCOPE_COMPLETION
import os


def parse_args():
    parser=argparse.ArgumentParser(description="a script to evaluate public models")
    parser.add_argument('--model_name_or_path', type=str, default=None,
                        help='Name/path of the model we are working with')
    parser.add_argument('--tokenizer_name', type=str, default='',
                        help='tokenizer name/path')
    parser.add_argument("--max-memory-per-gpu", type=str, 
                        help="Defines maximum memory allocated to gpu", default='28GB')
    parser.add_argument('--max_length', type=int, default=2048,
                        help='maximum total length')
    parser.add_argument('--max_context_length', type=int, default=1792,
                        help='maximum prompt length')
    parser.add_argument('--do_sample', default=False, action='store_true',
                        help='Sample / Greedy generation')
    parser.add_argument('--temperature', type=float, default=0.4,
                        help='temperature')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='top p')
    parser.add_argument('--top_k', type=int, default=200,
                        help='top k')
    parser.add_argument('--num_beams', type=int, default=1,
                        help='number of beams')
    parser.add_argument('--num_samples_per_example', type=int, default=1,
                        help='number of samples')
    parser.add_argument('--run_eval_only', default=False, action='store_true',
                        help='Run debugging part of the code')
    parser.add_argument('--bf16', default=False, action='store_true',
                        help='To use brain float 16')
    parser.add_argument('--fp16', default=False, action='store_true',
                        help='To use float 16')
    parser.add_argument('--use_fast_tokenizer', default=False, action='store_true',
                        help='Set to use fast tokenizer')
    parser.add_argument('--use_stopping_criteria', default=False, action='store_true',
                        help='Flag to use stopping criterion')
    parser.add_argument('--override_previous_results', default=False, action='store_true',
                        help='override previous results')
    parser.add_argument('--test_file', type=str, default='datasets/nominal/HumanEval.jsonl',
                        help='Test dataset')
    parser.add_argument('--programming_lang', type=str, default='python',
                        help='Programming Language')
    parser.add_argument('--valid_filter', type=str, default='func_ast_first',
                        help='Truncation logic')
    parser.add_argument('--output_dir', type=str, default='./eval_results/humaneval',
                        help='Folder to log the outputs')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='for torch.distributed')
    args=parser.parse_args()
    return args


def get_distributed_info():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    group_size = 1
    num_groups = world_size
    group = rank
    return {
        "local_rank": local_rank,
        "rank": rank,
        "group": group,
        "num_groups": num_groups,
        "group_size": group_size,
        "world_size": world_size,
    }


def count_files_present_nonemtpy(list_fname):
    count = 0
    for fname in list_fname:
        if os.path.isfile(fname):
            f = open(fname, "r", encoding="utf8")
            s = f.read()
            f.close()
            if not s == "":
                count += 1
    return count, len(list_fname)  


def execution_eval_generate(
    args,
    model,
    tokenizer,
    distributed_info
):
    device = distributed_info["local_rank"]
    if distributed_info["rank"] == 0:
        if not os.path.isdir(os.path.join(args.output_dir, "output")):
            os.makedirs(os.path.join(args.output_dir, "output"))
    
    # device = torch.device("cuda")
    problems = read_problems(args.test_file)
    num_samples = args.num_samples_per_example
    batch_size = 1
            
    fpath_format = os.path.join(
        args.output_dir, "output", "taskid-{task_idx}-gen{completion_idx}.json"
    )

    for enum_idx, task_id in enumerate(tqdm(problems)):
        # assume TaskName/ID format. The id part needs not be integer.
        task_idx = task_id.split("/")[1]
        if not (enum_idx % distributed_info["num_groups"] == distributed_info["group"]):
            continue
        if not args.override_previous_results:
            fnames = [
                fpath_format.format(task_idx=task_idx, completion_idx=_idx)
                for _idx in range(num_samples)
            ]
            count, all_count = count_files_present_nonemtpy(fnames)
            if count == all_count:
                print(
                    f"Result caching mode: Skipping case {task_id}. Generated all {all_count}"
                )
                continue
            else:
                print(
                    f"Result caching mode: Only {count} out of {all_count} were generated. Regenerating task {task_id}"
                )
        prompt = problems[task_id]["prompt"]
        if "language" in problems[task_id]:
            execution_language = problems[task_id]["language"]
        else:
            print("Warning -- no language in problem file")
            execution_language = None
        
        execution_prompt = None
        inputs = tokenizer(prompt)
        with torch.no_grad():
            if 'bloom' in args.model_name_or_path:
                input_ids = torch.tensor([inputs.input_ids[-args.max_context_length:-1]]).to(device)
            else:
                input_ids = torch.tensor([inputs.input_ids[-args.max_context_length:]]).to(device)

            completion_idx = -1
            for i in range(0, num_samples, batch_size):
                num_return_sequences = min(num_samples - i, batch_size)
                # one can save generation time by adding customized early stop criteria
                stopping_criteria = []

                output_dict = model.generate(
                    input_ids=input_ids,
                    max_length=args.max_length,
                    do_sample=args.do_sample,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    num_beams=args.num_beams,
                    num_return_sequences=num_return_sequences,
                    stopping_criteria=stopping_criteria,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                sequences = output_dict.sequences
                initial_context_length = len(sequences[0]) - len(output_dict.scores)

                predictions_post_eos = truncate(args, tokenizer, task_id, prompt, execution_prompt, input_ids, sequences, initial_context_length)

                for prediction in predictions_post_eos:
                    completion_idx += 1
                    fpath = fpath_format.format(
                        task_idx=task_idx, completion_idx=completion_idx
                    )
                    if execution_language is not None:
                        prediction["language"] = execution_language
                    with open(fpath, "w", encoding="utf8") as _f:
                        json.dump(prediction, _f)


def truncate(args, tokenizer, task_id, prompt, execution_prompt, input_ids, sequences, initial_context_length):
    """ Truncate the ori_pred to completion
    """
    if args.programming_lang == "python":
        predictions_post_eos = filter_valid_code(
                        true_str_input=prompt,
                        execution_prompt=execution_prompt,
                        inputs=input_ids,
                        sequences=sequences,
                        initial_context_length=initial_context_length,
                        tokenizer=tokenizer,
                        task_id=task_id,
                        post_process=args.valid_filter,
                        skip_special_tokens=True,
                        mean_logp=None,
                    )
    else:
        predictions_post_eos = inference_cut_off(
                        true_str_input=prompt,
                        inputs=input_ids,
                        sequences=sequences,
                        token_len_prompt_input=initial_context_length,
                        tokenizer=tokenizer,
                        skip_special_tokens=True,
                        task_id=task_id,
                        language=args.programming_lang,
                        input_indent=0,
                        mean_logp=None,
                    )
    return predictions_post_eos


def main():
    args=parse_args()
    distributed_info = get_distributed_info()
    if distributed_info["rank"] == 0:
        print('Arguments : ',  args)
    
    device = distributed_info["local_rank"] if torch.cuda.is_available() else 'cpu'
    if args.run_eval_only:
        if distributed_info["rank"] == 0:
            run_passatk_eval(
                args.test_file,
                args.programming_lang,
                args.output_dir,
                args.num_samples_per_example,
                args.override_previous_results,
            )
        return

    if 'codegen' in args.tokenizer_name:
        print('Setting pad token id')
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        tokenizer.pad_token_id = 50256
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=False)
        

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    
    if args.bf16:
        model.to(dtype=torch.bfloat16).to(device)
    elif args.fp16:
        model.to(dtype=torch.half).to(device)
    else:
        model.to(device)

    print(f'Loading {args.model_name_or_path} loading complete')
    
    distributed_info = get_distributed_info()
    execution_eval_generate(args, model, tokenizer, distributed_info)



if __name__ == '__main__':
    main()
