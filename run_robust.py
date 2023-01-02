""" This is the main python file to run our robustness benchmark.
Please run python run_robust.py --help to check detailed usage.

public models: [ "codegen-350M-multi", "codegen-2B-multi", "codegen-6B-multi",
    "codegen-350M-mono", "codegen-2B-mono", "codegen-6B-mono",
    "incoder-1B", "incoder-6B", 
    "gpt-j-6B",
    "codet5-base", "codet5-large"
]
Evaluated datasets: ["humaneval", "mbpp", "mbjp", "mbjsp", "mbphp", "mbrbp", "mbkp"]
"""

from __future__ import annotations
import os
import json
import argparse
# from config import *
import csv
import random
import numpy as np
from collections import Counter, defaultdict
from perturb import read_config

cwd = os.getcwd()


def run_cmd(cmd):
    """ A help function to run the command
    """
    print(f"=== {cmd} ===")
    os.system(cmd)


def read_json(file_name):
    """ A help funtion to load data json files
    """
    data = []
    if not os.path.exists(file_name):
        print(f"Warning: {file_name} not exists, skip!")
        return
    with open(file_name, 'r') as input_file:
        for line in input_file:
            data.append(json.loads(line))
    return data


def create_nominal_partial_datasets(args):
    """ create clean partial code for each dataset
    >>> example1 for creating nominal partial code: python run_robust.py create_partial natgen
    """
    for dataset in args.datasets:
        cmd1 = f"python perturb.py --data {dataset} --method natgen --task partial_code --create_partial_code"
        if args.overwrite: 
            cmd1 += " --overwrite"
        run_cmd(cmd1)


def create_perturbed_datasets(args):
    """ Perturbing the targeted datasets with different augmentation methods
    >>> example1 for nlaugmenter (e.g., with aug_method 1): python run_robust.py perturb nlaugmenter --aug_method 1 (--overwrite)
    >>> example2 for func_rename all aug_methods: python run_robust.py perturb func_name (--overwrite)
    >>> example3 for natgen code sturcture transformations with all aug_methods: python run_robust.py perturb natgen (--overwrite)
    >>> example4 for code format transformations all aug_methods: python run_robust.py perturb format (--overwrite)
    >>> example5 for a perturbed dataset with random perturbation selection: python run_robust.py perturb random --train (--overwrite)
    """
    for dataset in args.datasets:
        NL_AUG_RECIPES, PARTIAL_RECIPES, FUNC_RECIPES, FORMAT_RECIPES, FULL_RECIPES, RECIPES, \
                DATASET_PATH, RANDOM_TRANS, data_path, output_adv_path, model_generate_path, run_script = read_config(args.config, dataset)
        for seed in range(args.n_outputs):
            # if seed == 0: continue
            # if seed < 5: continue
            if args.method != "random":
                for aug_method in range(len(RECIPES[args.method])): # run each perturbing method
                    if args.aug_method is not None and aug_method != args.aug_method:
                        # specific args.aug_method index is given
                        continue
                    if RECIPES[args.method][aug_method] not in RANDOM_TRANS and seed >= 1: # skip other seeds since they are not in random
                        continue
                    # generate perturbation for the full dataset
                    cmd1 = f"python perturb.py --data {dataset} --subset full --method {args.method} --aug_method {aug_method} --seed {seed} --config {args.config}"
                    if args.overwrite: 
                        cmd1 += " --overwrite"
                    if args.method in ["format", "natgen"]:
                        cmd1 += " --task partial_code"
                    if args.print_sample:
                        cmd1 += " --print_sample"
                    run_cmd(cmd1)
                    # exit()
            else:
                cmd1 = f"python perturb.py --data {dataset} --subset full --method {args.method} --seed {seed}"
                if args.train:
                    cmd1 += " --train"
                if args.overwrite: 
                    cmd1 += " --overwrite"
                run_cmd(cmd1)


def evaluate_nominal(args):
    """ evaluate nominal results
    >>> example1 for regular dataset nominal evaluation: python run_robust.py nominal normal
    >>> example2 for partial code dataset nominal evaluation: python run_robust.py nominal natgen
    """
    eval_nominal = args.eval_only
    policy = "greedy" if args.num_samples == 0 else "sampling"
    for model in args.models:
        for dataset in args.datasets:
            NL_AUG_RECIPES, PARTIAL_RECIPES, FUNC_RECIPES, FORMAT_RECIPES, FULL_RECIPES, RECIPES, \
                DATASET_PATH, RANDOM_TRANS, data_path, output_adv_path, model_generate_path, run_script = read_config(args.config, dataset)
            if args.method not in ["natgen", "format"]: # run for regular dataset, not partial
                test_file = f"{cwd}/{data_path}/{DATASET_PATH[dataset]}"
                output_folder = f"{cwd}/{model_generate_path}/{model}/full/{dataset}/nominal"
                generated_sample_path = os.path.join(output_folder, f"{policy}/samples.jsonl")
                
                cmd1 = f"bash {run_script} {test_file} {output_folder} {dataset} {model} {args.ngpus} {args.overwrite} {args.num_samples}"
                cmd2 = f"evaluate_functional_correctness --sample_file {generated_sample_path} --problem_file {data_path}/{DATASET_PATH[dataset]}"
            else:
                test_file = f"{cwd}/{data_path}/{dataset + '_partial.jsonl'}"
                output_folder = f"{cwd}/{model_generate_path}/{model}/full/{dataset}_partial/nominal"
                generated_sample_path = os.path.join(output_folder, f"{policy}/samples.jsonl")
                
                cmd1 = f"bash {run_script} {test_file} {output_folder} {dataset} {model} {args.ngpus} {args.overwrite} {args.num_samples}"
                cmd2 = f"evaluate_functional_correctness --sample_file {generated_sample_path} --problem_file {data_path}/{dataset + '_partial.jsonl'}"
            
            if not eval_nominal:
                if not os.path.exists(generated_sample_path) or args.overwrite:
                    run_cmd(cmd1)
                else:
                    print(f"Warning: sample file {generated_sample_path} exists while overwrite is not enabled, skip!")
            else:
                if os.path.exists(generated_sample_path):
                    run_cmd(cmd2)
                else:
                    print(f"Warning: sample file {generated_sample_path} not exists!")
            print("\n")


def create_subset(args):
    """ [function deprecated!] create subset of perturbed dataset on the nominal correct prompts by specific model
    """
    # conda activate codex
    for model in args.models:
        for dataset in args.datasets:
            NL_AUG_RECIPES, PARTIAL_RECIPES, FUNC_RECIPES, FORMAT_RECIPES, FULL_RECIPES, RECIPES, \
                DATASET_PATH, RANDOM_TRANS, data_path, output_adv_path, model_generate_path, run_script = read_config(args.config, dataset)
            if args.method == "nlaugmenter":
                for seed in range(args.n_outputs):
                    for aug_method in range(len(NL_AUG_RECIPES)): # run each perturbing method
                        cmd2 = f"cd {args.aug_path} && python perturb.py --data {dataset} --model {model} --subset correct --method {args.method} --aug_method {aug_method} --seed {seed}"
                        if args.overwrite: 
                            cmd2 += " --overwrite"
                        run_cmd(cmd2)


def evaluate_perturbed_datasets(args):
    """ evaluate models on perturbed datasets
    >>> example1 for nlaugmenter (e.g., with aug_method 1): python run_robust.py exec nlaugmenter --aug_method 1 (--overwrite)
    >>> example2 for func_rename all aug_methods: python run_robust.py exec func_name (--overwrite)
    >>> example3 for natgen code sturcture transformations with all aug_methods: python run_robust.py exec natgen func_name (--overwrite)
    >>> example4 for code format transformations all aug_methods: python run_robust.py exec format func_name (--overwrite)
    """
    # evalset = "correct" # quick, only for pass_to_nonpass cases (deprecated)
    evalset = args.subset # slow, but can collect for nonpass_to_pass cases
    assert evalset == "full"
    method, eval_exec = args.method, args.eval_only
    policy = "greedy" if args.num_samples == 0 else "sampling"
    
    # using model to complete prompt and evaluate
    for model in args.models:
        for dataset in args.datasets:
            NL_AUG_RECIPES, PARTIAL_RECIPES, FUNC_RECIPES, FORMAT_RECIPES, FULL_RECIPES, RECIPES, \
                DATASET_PATH, RANDOM_TRANS, data_path, output_adv_path, model_generate_path, run_script = read_config(args.config, dataset)
            for aug_method in range(len(RECIPES[method])):
                if args.aug_method is not None and aug_method != args.aug_method:
                    # specific aug_method index is given
                    continue
                for seed in range(args.n_outputs):
                    if RECIPES[method][aug_method] not in RANDOM_TRANS and seed >= 1: # skip other seeds since they are not in random
                        continue
                    test_file = f"{cwd}/{output_adv_path}/{dataset}/{evalset}/{method}/{dataset}_{RECIPES[method][aug_method]}_s{seed}.jsonl"
                    output_folder = f"{cwd}/{model_generate_path}/{model}/{evalset}/{dataset}/{method}/{dataset}_{RECIPES[method][aug_method]}/s{seed}"
                    generated_sample_path = os.path.join(output_folder, f"{policy}/samples.jsonl")

                    cmd1 = f"bash {run_script} {test_file} {output_folder} {dataset} {model} {args.ngpus} {args.overwrite} {args.num_samples}"
                    cmd2 = f"evaluate_functional_correctness --sample_file {generated_sample_path} --problem_file {test_file}"
                    if not eval_exec:
                        if args.overwrite or not os.path.exists(generated_sample_path):
                            run_cmd(cmd1)
                        else:
                            # skip the generated ones
                            print(f"generated sample exists: {generated_sample_path}, skip...")
                    if eval_exec:
                        if os.path.exists(generated_sample_path):
                            run_cmd(cmd2)
                        else:
                            print(f"Warning: generated sample not exists, {generated_sample_path}")
                    print("\n")


def print_sample_analysis(args):
    """ retrieve each perturbed data and model completions to investigate samples individually
    stop at each prompt that nominal is correct but perturbed not
    >>> example1:  python run_robust.py analysis nlaugmenter --aug_method  0 --models None
    >>> example2:  python run_robust.py analysis format --aug_method 3 --models v5_672M_python --datasets mbpp
    """
    policy = "greedy" if args.num_samples == 0 else "sampling"
    assert args.aug_method is not None, "please assign --aug_method to analyze targeted perturbation samples"
    if args.models == ["None"]:
        # disable model generated sample checking, only evaluate for perturbed datasets
        model, dataset, aug_method, method = None, args.datasets[0], args.aug_method, args.method
    else:
        assert len(args.models) == 1 and len(args.datasets) == 1, "we only analyze samples for single model and single perturbed dataset"
        model, dataset, aug_method, method = args.models[0], args.datasets[0], args.aug_method, args.method
    
    NL_AUG_RECIPES, PARTIAL_RECIPES, FUNC_RECIPES, FORMAT_RECIPES, FULL_RECIPES, RECIPES, \
                DATASET_PATH, RANDOM_TRANS, data_path, output_adv_path, model_generate_path, run_script = read_config(args.config, args.datasets[0])

    perturbed_path = f"{output_adv_path}/{dataset}/full/{args.method}"

    data_orig, data_partial, data_perturbed = [], [], []
    data_orig = read_json(f"{data_path}/{DATASET_PATH[dataset]}") # original data file
    if args.method != "random":
        data_perturbed = read_json(f"{perturbed_path}/{dataset}_{RECIPES[args.method][aug_method]}_s{args.seed}.jsonl") # perturbed data file
    else:
        data_perturbed = read_json(f"{perturbed_path}/{dataset}_random_s{args.seed}.jsonl") # perturbed data file
    data = [data_orig, data_perturbed]

    if args.method in ["natgen", "format"]:
        data_partial = read_json(f"{data_path}/{dataset}_partial.jsonl")
        data.append(data_partial)
    
    if model is not None:
        # baseline nominal completion path
        if args.method == "format": # partial code completion
            # data_nominal_completed_path = f"{model_nominal_path}/{model}/full/{dataset}_partial/{policy}/samples.jsonl_results.jsonl"
            data_nominal_completed_path = f"{model_generate_path}/{model}/full/{dataset}_partial/nominal/{policy}/samples.jsonl_results.jsonl"
        elif args.method == "natgen": # black normalized partial code completion
            # data_nominal_completed_path = f"{model_generate_path}/{model}/full/{dataset}/natgen/{dataset}_Black/s0/{policy}/samples.jsonl_results.jsonl"
            data_nominal_completed_path = f"{model_generate_path}/{model}/full/{dataset}_partial/nominal/{policy}/samples.jsonl_results.jsonl"
        else: # regular dataset completion
            # data_nominal_completed_path = f"{model_nominal_path}/{model}/full/{dataset}/{policy}/samples.jsonl_results.jsonl"
            data_nominal_completed_path = f"{model_generate_path}/{model}/full/{dataset}/nominal/{policy}/samples.jsonl_results.jsonl"
        print("nominal completion path:", data_nominal_completed_path)
        data_nominal_completed = read_json(data_nominal_completed_path)
        data.append(data_nominal_completed)

        data_perturbed_completed_path = f"{model_generate_path}/{model}/full/{dataset}/{args.method}/{dataset}_{RECIPES[args.method][aug_method]}/s{args.seed}/{policy}/samples.jsonl_results.jsonl"
        # data_perturbed_completed_path = f"{model_generate_path}/{model}/full/{dataset}/{method}/{dataset}_{RECIPES[method][aug_method]}/s{args.seed}/{policy}_nostopping/samples.jsonl_results.jsonl"
        print("perturbed completion path:", data_perturbed_completed_path)
        data_perturbed_completed = read_json(data_perturbed_completed_path)
        data.append(data_perturbed_completed)

    # model != None & method in ["fomrat", "natgen"] [data_orig, data_perturbed, data_partial, data_nominal_completed, data_perturbed_completed]
    # model != None & method not in ["fomrat", "natgen"] [data_orig, data_perturbed, data_nominal_completed, data_perturbed_completed]
    # model == None [data_orig, data_perturbed]
    for entry in zip(*data):
        # if entry2["task_id"] != "MBPP/600": continue
        print("\n ===", entry[0]["task_id"], "===\n")
        assert entry[0]["task_id"] == entry[1]["task_id"], "task_id mis-match, wrongly perturbed dataset!"
        if args.method != "random":
            perturbed_aug_method = RECIPES[args.method][aug_method]
        else:
            perturbed_aug_method = entry[-1]["aug_method"]

        if method in ["natgen", "format"]:
            print(f"[nominal partial]\n{entry[2]['prompt']}")
            print(f"[perturbed partial ({perturbed_aug_method})]\n{entry[1]['prompt']}")
        else:
            print(f"[nominal]\n{entry[0]['prompt']}")
            print(f"[perturbed ({perturbed_aug_method})]\n{entry[1]['prompt']}")

        if model is not None and not entry[-1]["passed"] and entry[-2]["passed"]:
            # only stop at wrongly predicted samples
            print(f"[{model} nominal completion] (passed: {entry[-2]['passed']})\n{entry[-2]['input'] + entry[-2]['completion']}")
            print(f"[{model} perturbed [{RECIPES[method][aug_method]}] completion (passed: {entry[-1]['passed']})]\n{entry[-1]['input'] + entry[-1]['completion']}")
            import pdb; pdb.set_trace()

        if model is None:
            import pdb; pdb.set_trace()
    return


def read_passatk(file):
    f = open(file, "r")
    line = f.readlines()[0].replace("\'", "\"")
    data = json.loads(line)
    f.close()
    return data["pass@1"]


def calculate_passatk(data):
    length = len(data)
    cnt = 0
    for d in data:
        if d["passed"]:
            cnt += 1
    return cnt / length


def estimator(n, c, k):
    # calculate estimated passatk for each input problem
    if n - c < k:
        return 1.
    return 1. - np.prod(1. - k/np.arange(n - c + 1, n + 1))


def calculate_passatk_sampling(data, n=1, k=1):
    completion_id = Counter()
    n_samples = 0
    results = defaultdict(list)

    for d in data:
        task_id = d["task_id"]
        results[task_id].append([completion_id[task_id], d["passed"]])
        completion_id[task_id] += 1
        n_samples += 1
    
    single_passatk_list = []
    for task_id in results:
        if len(results) * n == len(data):
            c = sum(d[1] for d in results[task_id])
        else:
            assert n <= len(results[task_id])
            c = sum(results[task_id][ni][1] for ni in range(n))
        single_passatk_list.append(estimator(n, c, k))
    return sum(single_passatk_list) / len(single_passatk_list)


def read_into_dict(data):
    data_dict = {}
    for d in data:
        data_dict[d["task_id"]] = d["passed"]
    return data_dict


def get_worst_passatk_dict(perturbed_data_list):
    assert len(perturbed_data_list) >= 1
    passatk_worst = {}
    for pdata in perturbed_data_list[0]:
        passatk_worst[pdata["task_id"]] = True
    for perturbed_data in perturbed_data_list:
        for pdata in perturbed_data:
            assert pdata["task_id"] in passatk_worst
            passatk_worst[pdata["task_id"]] = passatk_worst[pdata["task_id"]] and pdata["passed"]
    return passatk_worst

def get_worst_passatk_dict_sampling(perturbed_data_list):
    assert len(perturbed_data_list) >= 1
    passatk_worst = defaultdict(list)
    completion_id = Counter()
    for pdata in perturbed_data_list[0]:
        task_id = pdata["task_id"]
        passatk_worst[task_id].append([completion_id[task_id], True])
        completion_id[task_id] += 1
    for perturbed_data in perturbed_data_list:
        completion_id = Counter()
        for pdata in perturbed_data:
            task_id = pdata["task_id"]
            assert task_id in passatk_worst
            passatk_worst[task_id][completion_id[task_id]][1] = passatk_worst[task_id][completion_id[task_id]][1] and pdata["passed"]
            completion_id[task_id] += 1
    return passatk_worst


def get_best_passatk_dict(perturbed_data_list):
    assert len(perturbed_data_list) >= 1
    passatk_best = {}
    for pdata in perturbed_data_list[0]:
        passatk_best[pdata["task_id"]] = False
    for perturbed_data in perturbed_data_list:
        for pdata in perturbed_data:
            assert pdata["task_id"] in passatk_best
            passatk_best[pdata["task_id"]] = passatk_best[pdata["task_id"]] or pdata["passed"]
    return passatk_best

def get_best_passatk_dict_sampling(perturbed_data_list):
    assert len(perturbed_data_list) >= 1
    passatk_best = defaultdict(list)
    completion_id = Counter()
    for pdata in perturbed_data_list[0]:
        task_id = pdata["task_id"]
        passatk_best[task_id].append([completion_id[task_id], False])
        completion_id[task_id] += 1
    for perturbed_data in perturbed_data_list:
        completion_id = Counter()
        for pdata in perturbed_data:
            task_id = pdata["task_id"]
            assert task_id in passatk_best
            passatk_best[task_id][completion_id[task_id]][1] = passatk_best[task_id][completion_id[task_id]][1] or pdata["passed"]
            completion_id[task_id] += 1
    return passatk_best


def calculate_metric(perturbed_data_list, metric, nominal_data):
    """ Get targeted metric numbers
    perturbed_data_list: a list of perturbed data completions, each element is the completion of one seed dataset
    """
    length = len(nominal_data)
    # init worst dict
    # passatk_worst = {}
    # for ndata in nominal_data:
    #     passatk_worst[ndata["task_id"]] = True
    passatk_worst = get_worst_passatk_dict(perturbed_data_list)
    passatk_best = get_best_passatk_dict(perturbed_data_list)
    if metric == "passatk":
        # perturbed pass@k
        passatk_list = []
        for perturbed_data in perturbed_data_list:
            passatk_list.append(calculate_passatk(perturbed_data))
        worst_cnt = 0
        for key in passatk_worst:
            if passatk_worst[key]: 
                worst_cnt += 1
        return passatk_list, worst_cnt / length if passatk_list else " ", passatk_worst

    if metric == "drop":
        # (nominal pass@k - perturbed pass@k) / nominal pass@k
        nominal_passatk = calculate_passatk(nominal_data)
        passatk_list = []
        for perturbed_data in perturbed_data_list:
            perturbed_passatk = calculate_passatk(perturbed_data)
            passatk_list.append((nominal_passatk - perturbed_passatk) / nominal_passatk)
        worst_cnt = 0
        for key in passatk_worst:
            if passatk_worst[key]: 
                worst_cnt += 1
        perturbed_passatk_worst = worst_cnt / length
        return passatk_list, (nominal_passatk - perturbed_passatk_worst) / nominal_passatk if passatk_list else " ", passatk_worst

    if metric == "relative":
        # (nominal != perturbed) / total prompts
        diffset = []
        nominal_dict = {}
        for ndata in nominal_data:
            nominal_dict[ndata["task_id"]] = ndata["passed"]
        relative_list = []
        for perturbed_data in perturbed_data_list:
            relative_cnt = 0
            for pdata in perturbed_data:
                if nominal_dict[pdata["task_id"]] != pdata["passed"]:
                    relative_cnt += 1
                    diffset.append(pdata["task_id"])
            relative_list.append(relative_cnt / length)
        diffset = set(diffset)
        worst_cnt = 0
        for key in passatk_worst:
            if nominal_dict[key] != passatk_worst[key]:
                worst_cnt += 1
            elif nominal_dict[key] != passatk_best[key]:
                worst_cnt += 1
        assert len(diffset) == worst_cnt
        return relative_list, worst_cnt / length  if relative_list else " ", passatk_worst

    if metric == "attack_success":
        # (nominal correct & perturbed incorrect) / nominal correct
        nominal_dict = {}
        correct_cnt = 0
        for ndata in nominal_data:
            nominal_dict[ndata["task_id"]] = ndata["passed"]
            if ndata["passed"]:
                correct_cnt += 1
        success_list = []
        for perturbed_data in perturbed_data_list:
            success_cnt = 0
            for pdata in perturbed_data:
                if nominal_dict[pdata["task_id"]] and not pdata["passed"]:
                    success_cnt += 1
            success_list.append(success_cnt / correct_cnt)
        worst_cnt = 0
        for key in passatk_worst:
            if nominal_dict[key] and not passatk_worst[key]:
                worst_cnt += 1
        return success_list, worst_cnt / correct_cnt  if success_list else " ", passatk_worst


def report_results(args):
    """ report all the nominal and perturbed completion results by the models
    >>> example1: python run_robust.py report nlaugmenter --models codegen-350M-mono codegen-350M-multi --datasets humaneval mbpp --n_outputs 5
    >>> example2: python run_robust.py report nlaugmenter --aug_method 0 --models codegen-350M-mono codegen-350M-multi --datasets humaneval mbpp --n_outputs 5 --metric drop
    >>> example3: python run_robust.py report natgen --models codegen-350M-mono codegen-350M-multi codegen-2B-mono codegen-2B-multi codegen-6B-mono codegen-6B-multi codegen-16B-mono codegen-16B-multi gpt-j-6B incoder-1B incoder-6B --datasets humaneval mbpp --metric passatk
    """
    policy = "greedy" if args.num_samples == 0 else "sampling"
    results = {} # save all the results for saving csv table
    for dataset in args.datasets:
        results[dataset] = {}
        for model in args.models:
            results[dataset][model] = {}
    for model in args.models:
        for dataset in args.datasets:
            NL_AUG_RECIPES, PARTIAL_RECIPES, FUNC_RECIPES, FORMAT_RECIPES, FULL_RECIPES, RECIPES, \
                DATASET_PATH, RANDOM_TRANS, data_path, output_adv_path, model_generate_path, run_script = read_config(args.config, dataset)
            print(f"[{model}, {dataset}]")
            # baseline nominal completion path
            if args.method == "format": # partial code completion
                data_nominal_completed_path = f"{model_generate_path}/{model}/full/{dataset}_partial/nominal/{policy}/samples.jsonl_results.jsonl"
            elif args.method == "natgen": # black normalized partial code completion
                # data_nominal_completed_path = f"{model_generate_path}/{model}/full/{dataset}/natgen/{dataset}_Black/s0/{policy}/samples.jsonl_results.jsonl"
                data_nominal_completed_path = f"{model_generate_path}/{model}/full/{dataset}_partial/nominal/{policy}/samples.jsonl_results.jsonl"
            else: # regular dataset completion
                data_nominal_completed_path = f"{model_generate_path}/{model}/full/{dataset}/nominal/{policy}/samples.jsonl_results.jsonl"
            # nominal_passatk = read_passatk(data_nominal_completed_path)
            if not os.path.exists(data_nominal_completed_path):
                print(f"{data_nominal_completed_path} missing, skip...")
                continue
            data_nominal_completed = read_json(data_nominal_completed_path)
            nominal_passatk = calculate_passatk(data_nominal_completed)
            print(f"nominal pass@1: {nominal_passatk:.4f}")
            results[dataset][model]["nominal"] = nominal_passatk

            for aug_method in range(len(RECIPES[args.method])):
                if args.aug_method is not None and aug_method != args.aug_method:
                    # specific aug_method index is given
                    continue
                perturbed_data_list = []
                for seed in range(args.n_outputs):
                    if RECIPES[args.method][aug_method] not in RANDOM_TRANS and seed >= 1: # skip other seeds since they are not in random
                        continue
                    data_perturbed_completed_path = f"{model_generate_path}/{model}/full/{dataset}/{args.method}/{dataset}_{RECIPES[args.method][aug_method]}/s{seed}/{policy}/samples.jsonl_results.jsonl"
                    # print(data_perturbed_completed_path)
                    if os.path.exists(data_perturbed_completed_path):
                        perturbed_data_list.append(read_json(data_perturbed_completed_path))
                    else:
                        print(f"{data_perturbed_completed_path} not exists, skip..")
                        # import pdb; pdb.set_trace()
                        pass
                passatk_list, passatk_worst, _ = calculate_metric(perturbed_data_list, args.metric, data_nominal_completed)
                if passatk_list:
                    print(f"\t{RECIPES[args.method][aug_method]} {args.metric}: {passatk_list}, {passatk_worst:.4f}")
                else:
                    print(f"\t{RECIPES[args.method][aug_method]} {args.metric}: {passatk_list}, {passatk_worst}")
                # print(f"\t{RECIPES[args.method][aug_method]} {args.metric}: {passatk_worst:.4f}")
                results[dataset][model][RECIPES[args.method][aug_method]] = passatk_worst

    # reformulate results to csv table
    for dataset in args.datasets:
        NL_AUG_RECIPES, PARTIAL_RECIPES, FUNC_RECIPES, FORMAT_RECIPES, FULL_RECIPES, RECIPES, \
                DATASET_PATH, RANDOM_TRANS, data_path, output_adv_path, model_generate_path, run_script = read_config(args.config, dataset)
        full_data = []
        row = ["nominal"]
        for model in args.models:
            if "nominal" not in results[dataset][model]:
                row.append(" ")
            else:
                row.append(results[dataset][model]["nominal"])
        full_data.append(row)
        for aug_method in range(len(RECIPES[args.method])):
            row = [RECIPES[args.method][aug_method]]
            if args.aug_method is not None and aug_method != args.aug_method:
                # specific aug_method index is given
                continue
            for model in args.models:
                if RECIPES[args.method][aug_method] not in results[dataset][model]:
                    row.append(" ")
                else:
                    row.append(results[dataset][model][RECIPES[args.method][aug_method]])
            full_data.append(row)

        header = [args.metric] + args.models
        csv_path = f"csv/{dataset}_{args.method}_{args.metric}.csv"
        if not os.path.exists("csv"):
            os.mkdir("csv")
        file = open(csv_path, "w")
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(full_data)
        file.close()
    return


def report_results_coarse(args):
    """ report all the nominal and perturbed completion results by the models across the same perturbation category
    >>> example1: python run_robust.py report_coarse natgen --models codegen-350M-mono codegen-350M-multi codegen-6B-mono codegen-6B-multi incoder-1B incoder-6B gpt-j-6B --datasets humaneval mbpp --n_outputs 5
    """
    policy = "greedy" if args.num_samples == 0 else "sampling"
    results = {} # save all the perturbed dict for saving csv table [worst_dict, best_dict]
    nominal_dict = {} # save nominal data dict 
    nominal_dict_passatk = {} # save nominal passatk
    for dataset in args.datasets:
        results[dataset] = {}
        nominal_dict[dataset] = {}
        nominal_dict_passatk[dataset] = {}
        for model in args.models:
            results[dataset][model] = {}
            nominal_dict[dataset][model] = {}
            nominal_dict_passatk[dataset][model] = {}
    for model in args.models:
        for dataset in args.datasets:
            NL_AUG_RECIPES, PARTIAL_RECIPES, FUNC_RECIPES, FORMAT_RECIPES, FULL_RECIPES, RECIPES, \
                DATASET_PATH, RANDOM_TRANS, data_path, output_adv_path, model_generate_path, run_script = read_config(args.config, dataset)
            print(f"[{model}, {dataset}]")
            # baseline nominal completion path
            if args.method == "format": # partial code completion
                # data_nominal_completed_path = f"{model_generate_path}/{model}/full/{dataset}_partial/nominal/{policy}/samples.jsonl_passatk.txt"
                data_nominal_completed_path = f"{model_generate_path}/{model}/full/{dataset}_partial/nominal/{policy}/samples.jsonl_results.jsonl"
            elif args.method == "natgen": # black normalized partial code completion
                # data_nominal_completed_path = f"{model_generate_path}/{model}/full/{dataset}/natgen/{dataset}_Black/s0/{policy}/samples.jsonl_results.jsonl"
                data_nominal_completed_path = f"{model_generate_path}/{model}/full/{dataset}_partial/nominal/{policy}/samples.jsonl_results.jsonl"
            else: # regular dataset completion
                data_nominal_completed_path = f"{model_generate_path}/{model}/full/{dataset}/nominal/{policy}/samples.jsonl_results.jsonl"
            # nominal_passatk = read_passatk(data_nominal_completed_path)
            if not os.path.exists(data_nominal_completed_path):
                print(f"{data_nominal_completed_path} missing, skip...")
                continue
            data_nominal_completed = read_json(data_nominal_completed_path)
            if policy == "greedy":
                nominal_passatk = calculate_passatk(data_nominal_completed)
            else:
                nominal_passatk = calculate_passatk_sampling(data_nominal_completed, n=args.num_samples, k=args.k)
            if policy == "greedy":
                nominal_data_dict = read_into_dict(data_nominal_completed)
            else:
                nominal_data_dict = get_worst_passatk_dict_sampling([data_nominal_completed])
            print(f"nominal pass@1: {nominal_passatk:.4f}")
            nominal_dict[dataset][model] = nominal_data_dict
            nominal_dict_passatk[dataset][model] = nominal_passatk

            results[dataset][model] = None
            for aug_method in range(len(RECIPES[args.method])):
                if args.aug_method is not None and aug_method != args.aug_method:
                    # specific aug_method index is given
                    continue
                perturbed_data_list = []
                for seed in range(args.n_outputs):
                    if RECIPES[args.method][aug_method] not in RANDOM_TRANS and seed >= 1: # skip other seeds since they are not in random
                        continue
                    data_perturbed_completed_path = f"{model_generate_path}/{model}/full/{dataset}/{args.method}/{dataset}_{RECIPES[args.method][aug_method]}/s{seed}/{policy}/samples.jsonl_results.jsonl"
                    # print(data_perturbed_completed_path)
                    if os.path.exists(data_perturbed_completed_path):
                        perturbed_data_list.append(read_json(data_perturbed_completed_path))
                    else:
                        print(f"{data_perturbed_completed_path} not exists, skip..")
                        pass
                passatk_list, passatk_worst, _ = calculate_metric(perturbed_data_list, "passatk", data_nominal_completed)
                if policy == "greedy":
                    passatk_worst_dict = get_worst_passatk_dict(perturbed_data_list)
                    passatk_best_dict = get_best_passatk_dict(perturbed_data_list)
                else:
                    # passatk_worst_dict[task_id] = [[completion_id, True/False]]
                    passatk_worst_dict = get_worst_passatk_dict_sampling(perturbed_data_list)
                    passatk_best_dict = get_best_passatk_dict_sampling(perturbed_data_list)
                if passatk_list:
                    print(f"\t{RECIPES[args.method][aug_method]} passatk: {passatk_list}, {passatk_worst:.4f}")
                    # import pdb; pdb.set_trace()
                    # merge results across different aug_method
                    if results[dataset][model] is None:
                        results[dataset][model] = [passatk_worst_dict, passatk_best_dict]
                    else:
                        for key in results[dataset][model][0]:
                            assert key in passatk_worst_dict and key in passatk_best_dict
                            if policy == "greedy":
                                results[dataset][model][0][key] = results[dataset][model][0][key] and passatk_worst_dict[key]
                                results[dataset][model][1][key] = results[dataset][model][1][key] or passatk_best_dict[key]
                            else:
                                for completion_id in range(len(results[dataset][model][0][key])):
                                    results[dataset][model][0][key][completion_id][1] = results[dataset][model][0][key][completion_id][1] and passatk_worst_dict[key][completion_id][1]
                                    results[dataset][model][1][key][completion_id][1] = results[dataset][model][1][key][completion_id][1] and passatk_best_dict[key][completion_id][1]
                else:
                    # no data available
                    print(f"\t{RECIPES[args.method][aug_method]} passatk: {passatk_list}, {passatk_worst}")

    json.dump(nominal_dict, open(f"statitic_jsons/{args.method}_nominal.json", "w"))
    json.dump(results, open(f"statitic_jsons/{args.method}_perturbed.json", "w"))
    # json.load(nominal_dict, open(f"statitic_jsons/{args.method}_nominal.json", "r"))

    # reformulate results to csv table
    for dataset in args.datasets:
        NL_AUG_RECIPES, PARTIAL_RECIPES, FUNC_RECIPES, FORMAT_RECIPES, FULL_RECIPES, RECIPES, \
                DATASET_PATH, RANDOM_TRANS, data_path, output_adv_path, model_generate_path, run_script = read_config(args.config, dataset)
        full_data = []
        row = ["nominal"]
        for model in args.models:
            if dataset in nominal_dict_passatk and model in nominal_dict_passatk[dataset]:
                row.append(nominal_dict_passatk[dataset][model])
            else:
                row.append(" ")
        full_data.append(row)
        
        row = ["passatk"]
        for model in args.models:
            cnt = 0
            total_cnt = 0
            if results[dataset][model][0]: # worst dict
                if policy == "greedy":
                    for key in results[dataset][model][0]:
                        if results[dataset][model][0][key]:
                            cnt += 1
                        total_cnt += 1
                    row.append(cnt / total_cnt)
                else:
                    single_passatk_list = []
                    for task_id in results[dataset][model][0]:
                        assert args.num_samples <= len(results[dataset][model][0][task_id])
                        c = sum(results[dataset][model][0][task_id][ni][1] for ni in range(args.num_samples))
                        # c = sum(d[1] for d in results[dataset][model][0][task_id])
                        single_passatk_list.append(estimator(args.num_samples, c, args.k))
                    row.append(sum(single_passatk_list) / len(single_passatk_list))
            else:
                row.append(" ")
        full_data.append(row)

        row = ["drop (%)"]
        for model in args.models:
            cnt = 0
            total_cnt = 0
            if results[dataset][model][0]:
                if policy == "greedy":
                    for key in results[dataset][model][0]:
                        if results[dataset][model][0][key]:
                            cnt += 1
                        total_cnt += 1
                    passatk = cnt / total_cnt
                else:
                    single_passatk_list = []
                    for task_id in results[dataset][model][0]:
                        assert args.num_samples <= len(results[dataset][model][0][task_id])
                        c = sum(results[dataset][model][0][task_id][ni][1] for ni in range(args.num_samples))
                        # c = sum(d[1] for d in results[dataset][model][0][task_id])
                        single_passatk_list.append(estimator(args.num_samples, c, args.k))
                    passatk = sum(single_passatk_list) / len(single_passatk_list)
                nominal_passatk = nominal_dict_passatk[dataset][model]
                row.append((nominal_passatk - passatk) / nominal_passatk * 100.)
            else:
                row.append(" ")
        full_data.append(row)
        
        row = ["relative (%)"]
        for model in args.models:
            cnt = 0
            total_cnt = 0
            if results[dataset][model][0]:
                for task_id in results[dataset][model][0]:
                    if policy == "greedy":
                        if results[dataset][model][0][task_id] != nominal_dict[dataset][model][task_id]:
                            # worst dict difference
                            cnt += 1
                        elif results[dataset][model][1][task_id] != nominal_dict[dataset][model][task_id]:
                            # best dict difference
                            cnt += 1
                    else:
                        # assert args.num_samples == len(results[dataset][model][0][task_id])
                        posc, negc = 0, 0 # for each task_id, how many are pos changes/neg changes out of n 
                        for completion_id in range(args.num_samples):
                            if results[dataset][model][0][task_id][completion_id][1] != nominal_dict[dataset][model][task_id][completion_id][1]:
                                # worst dict difference
                                negc += 1
                            elif results[dataset][model][1][task_id][completion_id][1] != nominal_dict[dataset][model][task_id][completion_id][1]:
                                # best dict difference
                                posc += 1
                        # estimate the probability of pos change/neg change for each input with sampling k out of n
                        cnt += estimator(args.num_samples, posc, args.k) + estimator(args.num_samples, negc, args.k)
                    total_cnt += 1
                row.append(cnt / total_cnt * 100.)
            else:
                row.append(" ")
        full_data.append(row)

        header = [args.method] + args.models
        csv_path = f"csv_coarse/{dataset}_{args.method}.csv"
        if not os.path.exists("csv_coarse"):
            os.mkdir("csv_coarse")
        file = open(csv_path, "w")
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(full_data)
        file.close()
    return


def report_results_finegrained(args):
    """ report all the nominal and perturbed completion results by the models across the same perturbation category
    >>> example1: python run_robust.py report_finegrained natgen --models codegen-350M-mono codegen-350M-multi codegen-6B-mono codegen-6B-multi incoder-1B incoder-6B gpt-j-6B --datasets humaneval mbpp --n_outputs 5
    """
    policy = "greedy" if args.num_samples == 0 else "sampling"
    results = {} # save all the perturbed dict for saving csv table [worst_dict, best_dict]
    nominal_dict = {} # save nominal data dict 
    nominal_dict_passatk = {} # save nominal passatk
    for dataset in args.datasets:
        results[dataset] = {}
        nominal_dict[dataset] = {}
        nominal_dict_passatk[dataset] = {}
        for model in args.models:
            results[dataset][model] = {}
            nominal_dict[dataset][model] = {}
            nominal_dict_passatk[dataset][model] = {}
    for model in args.models:
        for dataset in args.datasets:
            NL_AUG_RECIPES, PARTIAL_RECIPES, FUNC_RECIPES, FORMAT_RECIPES, FULL_RECIPES, RECIPES, \
                DATASET_PATH, RANDOM_TRANS, data_path, output_adv_path, model_generate_path, run_script = read_config(args.config, dataset)
            print(f"[{model}, {dataset}]")
            # baseline nominal completion path
            if args.method == "format": # partial code completion
                # data_nominal_completed_path = f"{model_generate_path}/{model}/full/{dataset}_partial/nominal/{policy}/samples.jsonl_passatk.txt"
                data_nominal_completed_path = f"{model_generate_path}/{model}/full/{dataset}_partial/nominal/{policy}/samples.jsonl_results.jsonl"
            elif args.method == "natgen": # black normalized partial code completion
                # data_nominal_completed_path = f"{model_generate_path}/{model}/full/{dataset}/natgen/{dataset}_Black/s0/{policy}/samples.jsonl_results.jsonl"
                data_nominal_completed_path = f"{model_generate_path}/{model}/full/{dataset}_partial/nominal/{policy}/samples.jsonl_results.jsonl"
            else: # regular dataset completion
                data_nominal_completed_path = f"{model_generate_path}/{model}/full/{dataset}/nominal/{policy}/samples.jsonl_results.jsonl"
            # nominal_passatk = read_passatk(data_nominal_completed_path)
            if not os.path.exists(data_nominal_completed_path):
                print(f"{data_nominal_completed_path} missing, skip...")
                continue
            data_nominal_completed = read_json(data_nominal_completed_path)
            nominal_passatk = calculate_passatk(data_nominal_completed)
            nominal_data_dict = read_into_dict(data_nominal_completed)
            print(f"nominal pass@1: {nominal_passatk:.4f}")
            nominal_dict[dataset][model] = nominal_data_dict
            nominal_dict_passatk[dataset][model] = nominal_passatk

            for aug_method in range(len(RECIPES[args.method])):
                if args.aug_method is not None and aug_method != args.aug_method:
                    # specific aug_method index is given
                    continue
                perturbed_data_list = []
                for seed in range(args.n_outputs):
                    if RECIPES[args.method][aug_method] not in RANDOM_TRANS and seed >= 1: # skip other seeds since they are not in random
                        continue
                    data_perturbed_completed_path = f"{model_generate_path}/{model}/full/{dataset}/{args.method}/{dataset}_{RECIPES[args.method][aug_method]}/s{seed}/{policy}/samples.jsonl_results.jsonl"
                    # print(data_perturbed_completed_path)
                    if os.path.exists(data_perturbed_completed_path):
                        perturbed_data_list.append(read_json(data_perturbed_completed_path))
                    else:
                        # print(f"{data_perturbed_completed_path} not exists, skip..")
                        pass
                passatk_list, passatk_worst, _ = calculate_metric(perturbed_data_list, "passatk", data_nominal_completed)
                _, drop_worst, _ = calculate_metric(perturbed_data_list, "drop", data_nominal_completed)
                _, relative_worst, _ = calculate_metric(perturbed_data_list, "relative", data_nominal_completed)
                print(f"\t{RECIPES[args.method][aug_method]} passatk: {passatk_list}, {passatk_worst:.4f}")
                if passatk_list:
                    results[dataset][model][RECIPES[args.method][aug_method]] = [passatk_worst, drop_worst, relative_worst]
                else:
                    results[dataset][model][RECIPES[args.method][aug_method]] = None

    # reformulate results to csv table
    for dataset in args.datasets:
        NL_AUG_RECIPES, PARTIAL_RECIPES, FUNC_RECIPES, FORMAT_RECIPES, FULL_RECIPES, RECIPES, \
                DATASET_PATH, RANDOM_TRANS, data_path, output_adv_path, model_generate_path, run_script = read_config(args.config, dataset)
        full_data = []
        row = ["nominal", "passatk"]
        for model in args.models:
            if dataset in nominal_dict_passatk and model in nominal_dict_passatk[dataset]:
                row.append(f"{nominal_dict_passatk[dataset][model]:.3f}")
            else:
                row.append(" ")
        full_data.append(row)

        for aug_method in range(len(RECIPES[args.method])):

            row = [RECIPES[args.method][aug_method], "passatk"]
            for model in args.models:
                if results[dataset][model][RECIPES[args.method][aug_method]]:
                    row.append(f"{results[dataset][model][RECIPES[args.method][aug_method]][0]:.3f}")
                else:
                    row.append(" ")
            full_data.append(row)

            row = [RECIPES[args.method][aug_method], "drop (%)"]
            for model in args.models:
                if results[dataset][model][RECIPES[args.method][aug_method]]:
                    row.append(f"{results[dataset][model][RECIPES[args.method][aug_method]][1]*100.:.2f}")
                else:
                    row.append(" ")
            full_data.append(row)
            
            row = [RECIPES[args.method][aug_method], "relative (%)"]
            for model in args.models:
                if results[dataset][model][RECIPES[args.method][aug_method]]:
                    row.append(f"{results[dataset][model][RECIPES[args.method][aug_method]][2]*100.:.2f}")
                else:
                    row.append(" ")
            full_data.append(row)

        header = ["Perturbations", "Metric"] + args.models
        csv_path = f"csv_finegrained/{dataset}_{args.method}.csv"
        if not os.path.exists("csv_finegrained"):
            os.mkdir("csv_finegrained")
        file = open(csv_path, "w")
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(full_data)
        file.close()
    return


if __name__ == '__main__':
    """ The main function for using our robustness benchmark
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('status', type=str, choices=['perturb', 'create_partial', 'nominal', 'subset', 'exec', 'analysis', 'report', "report_coarse", "report_finegrained"], help='The funcitons enabled by our benchmark')
    parser.add_argument('method', type=str, choices=["normal", "nlaugmenter", "natgen", "format", "func_name", "random"], help="The classes of perturbation. Please set method to natgen with status nominal to evaluate nominal partial code.")
    parser.add_argument('--config', default="config.json", help="The config to run.")
    parser.add_argument('--aug_method', type=int, default=None, help="The detailed augmentation method used with index (index defined in config.json for each method). Default None means running all the perturbations")
    parser.add_argument('--datasets', nargs='+', default=["humaneval"], help='A list of datasets to perturb/evaluate with')
    parser.add_argument('--models', nargs='+', default=["codegen-350M-mono"], help="A list of the models needed to evaluate with (or create subset dataset for perturbed dataset, not needed most of the times).")
    parser.add_argument('--n_outputs', type=int, default=1, help="The total number of perturbations generated/evaluated with")
    parser.add_argument('--ngpus', type=int, default=1, help="The number of gpus to use.")
    parser.add_argument('--overwrite', action="store_true", help="Set overwrite to True if regenerate dataset perturbation or evaluation")
    parser.add_argument('--subset', type=str, default="full", choices=["full", "correct", "incorrect"], help="Using the whole dataset or only subsample for targeted models (deprecated).")
    parser.add_argument('--metric', type=str, default="passatk", choices=["passatk", "drop", "relative", "attack_success", "all"], help="The metric used for reporting results.")
    parser.add_argument('--eval_only', action="store_true", help="Only want to reevaluate model generated completions.")
    parser.add_argument('--rng-seed', type=int, default=42, help="global random seed.")
    parser.add_argument('--seed', type=int, default=0, help="Assign specific random seed for analysis option")
    parser.add_argument('--num_samples', type=int, default=0, help="Number of samples for predictions; Default 0 to be greedy.")
    parser.add_argument('--k', type=int, default=1, help="k in passatk; Number of trials allowed.")
    parser.add_argument('--print_sample', action="store_true", help="For debug purpose, print each perturbed sample with pdb stop.")
    args = parser.parse_args()
    assert args.status in ["nominal", "create_partial"] or args.method != "normal", "please specify perturbation method --method when --status is not nominal/create_partial!"
    print(args)
    
    if args.status == "nominal":
        evaluate_nominal(args)
    elif args.status == "create_partial":
        create_nominal_partial_datasets(args)
    elif args.status == "subset":
        create_subset(args)
    elif args.status == "perturb":
        create_perturbed_datasets(args)
    elif args.status == "exec":
        evaluate_perturbed_datasets(args)
    elif args.status == "analysis":
        print_sample_analysis(args)
    elif args.status == "report":
        if args.metric == "all":
            for metric in ["passatk", "drop", "relative"]:
                args.metric = metric
                report_results(args)
        else:
            report_results(args)
    elif args.status == "report_coarse":
        report_results_coarse(args)
    elif args.status == "report_finegrained":
        report_results_finegrained(args)


