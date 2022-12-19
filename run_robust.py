""" This is the main python file to run our robustness benchmark.
Please run python run_robust.py --help to check detailed usage.

Here is a list of models and datasets that have been evaluated
Evaluated private models: ["v5_672M_python", 
    "125M_multi", "125M_python", "125M_java", "125M_js", 
    "672M_multi", "672M_python", "672M_java", "672M_js", 
    "2_7B_multi", "2_7B_python", "2_7B_java", "2_7B_js", 
    "13B_multi", "13B_python", "13B_java", "13B_js", 
]
public models: [ "codegen-350M-multi", "codegen-2B-multi", "codegen-6B-multi",
    "codegen-350M-mono", "codegen-2B-mono", "codegen-6B-mono",
    "incoder-1B", "incoder-6B", 
    "gpt-j-6B",
    "codet5-base", "codet5-large"
]
Evaluated datasets: ["humaneval", "mbpp", "mbjp", "mbjsp", "mbphp", "mbrbp", "mbkp"]
"""

import os
import json
import argparse
from config import *
import csv


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
    >>> example3 for natgen code sturcture transformations with all aug_methods: python run_robust.py perturb natgen func_name (--overwrite)
    >>> example4 for code format transformations all aug_methods: python run_robust.py perturb format func_name (--overwrite)
    """
    for dataset in args.datasets:
        for seed in range(args.n_outputs):
            for aug_method in range(len(RECIPES[args.method])): # run each perturbing method
                if args.aug_method is not None and aug_method != args.aug_method:
                    # specific args.aug_method index is given
                    continue
                # generate perturbation for the full dataset
                cmd1 = f"python perturb.py --data {dataset} --subset full --method {args.method} --aug_method {aug_method} --seed {seed}"
                if args.overwrite: 
                    cmd1 += " --overwrite"
                if args.method in ["format", "natgen"]:
                    cmd1 += " --task partial_code"
                run_cmd(cmd1)
                # exit()


def evaluate_nominal(args):
    """ evaluate nominal results
    >>> example1 for regular dataset nominal evaluation: python run_robust.py nominal normal
    >>> example2 for partial code dataset nominal evaluation: python run_robust.py nominal natgen
    """
    eval_nominal = args.eval_only
    for model in args.models:
        for dataset in args.datasets:
            if args.method not in ["natgen", "format"]: # run for regular dataset, not partial
                test_file = f"{data_path}/{DATASET_PATH[dataset]}"
                # output_folder = f"{model_nominal_path}/{model}/full/{dataset}"
                output_folder = f"{model_generate_path}/{model}/full/{dataset}/nominal"
                generated_sample_path = os.path.join(output_folder, "greedy/samples.jsonl")
                
                cmd1 = f"bash {run_script} {test_file} {output_folder} {dataset} {model} {args.ngpus} {args.overwrite}"
                cmd2 = f"evaluate_functional_correctness {generated_sample_path} {data_path}/{DATASET_PATH[dataset]}"
            else:
                test_file = f"{data_path}/{dataset + '_partial.jsonl'}"
                # output_folder = f"{model_nominal_path}/{model}/full/{dataset}_partial"
                output_folder = f"{model_generate_path}/{model}/full/{dataset}_partial/nominal"
                generated_sample_path = os.path.join(output_folder, "greedy/samples.jsonl")
                
                cmd1 = f"bash {run_script} {test_file} {output_folder} {dataset} {model} {args.ngpus} {args.overwrite}"
                cmd2 = f"evaluate_functional_correctness {generated_sample_path} {data_path}/{dataset + '_partial.jsonl'}"
            
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
    
    # using model to complete prompt and evaluate
    for model in args.models:
        for dataset in args.datasets:
            for aug_method in range(len(RECIPES[method])):
                if args.aug_method is not None and aug_method != args.aug_method:
                    # specific aug_method index is given
                    continue
                for seed in range(args.n_outputs):
                    test_file = f"{output_adv_path}/{dataset}/{evalset}/{method}/{dataset}_{RECIPES[method][aug_method]}_s{seed}.jsonl"
                    output_folder = f"{model_generate_path}/{model}/{evalset}/{dataset}/{method}/{dataset}_{RECIPES[method][aug_method]}/s{seed}"
                    generated_sample_path = os.path.join(output_folder, "greedy/samples.jsonl")

                    cmd1 = f"bash {run_script} {test_file} {output_folder} {dataset} {model} {args.ngpus} {args.overwrite}"
                    cmd2 = f"evaluate_functional_correctness {generated_sample_path} {test_file}"
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


def print_sample_analysis(args, seed=0):
    """ retrieve each perturbed data and model completions to investigate samples individually
    stop at each prompt that nominal is correct but perturbed not
    >>> example1:  python run_robust.py analysis nlaugmenter --aug_method  0 --models None
    >>> example2:  python run_robust.py analysis format --aug_method 3 --models v5_672M_python --datasets mbpp
    """
    if args.models == ["None"]:
        # disable model generated sample checking, only evaluate for perturbed datasets
        model, dataset, aug_method, method = None, args.datasets[0], args.aug_method, args.method
    else:
        assert len(args.models) == 1 and len(args.datasets) == 1, "we only analyze samples for single model and single perturbed dataset"
        assert args.aug_method is not None, "please assign --aug_method to analyze targeted perturbation samples"
        model, dataset, aug_method, method = args.models[0], args.datasets[0], args.aug_method, args.method

    perturbed_path = f"{output_adv_path}/{dataset}/full/{args.method}"

    data_orig, data_partial, data_perturbed = [], [], []
    data_orig = read_json(f"{data_path}/{DATASET_PATH[dataset]}") # original data file
    data_perturbed = read_json(f"{perturbed_path}/{dataset}_{RECIPES[args.method][aug_method]}_s{seed}.jsonl") # perturbed data file
    data = [data_orig, data_perturbed]

    if args.method in ["natgen", "format"]:
        data_partial = read_json(f"{data_path}/{dataset}_partial.jsonl")
        data.append(data_partial)
    
    if model is not None:
        # baseline nominal completion path
        if args.method == "format": # partial code completion
            # data_nominal_completed_path = f"{model_nominal_path}/{model}/full/{dataset}_partial/greedy/samples.jsonl_results.jsonl"
            data_nominal_completed_path = f"{model_generate_path}/{model}/full/{dataset}_partial/nominal/greedy/samples.jsonl_results.jsonl"
        elif args.method == "natgen": # black normalized partial code completion
            data_nominal_completed_path = f"{model_generate_path}/{model}/full/{dataset}/natgen/{dataset}_Black/s0/greedy/samples.jsonl_results.jsonl"
        else: # regular dataset completion
            # data_nominal_completed_path = f"{model_nominal_path}/{model}/full/{dataset}/greedy/samples.jsonl_results.jsonl"
            data_nominal_completed_path = f"{model_generate_path}/{model}/full/{dataset}/nominal/greedy/samples.jsonl_results.jsonl"
        print("nominal completion path:", data_nominal_completed_path)
        data_nominal_completed = read_json(data_nominal_completed_path)
        data.append(data_nominal_completed)

        data_perturbed_completed_path = f"{model_generate_path}/{model}/full/{dataset}/{args.method}/{dataset}_{RECIPES[args.method][aug_method]}/s{seed}/greedy/samples.jsonl_results.jsonl"
        # data_perturbed_completed_path = f"{model_generate_path}/{model}/full/{dataset}/{method}/{dataset}_{RECIPES[method][aug_method]}/s{seed}/greedy_nostopping/samples.jsonl_results.jsonl"
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

        if method in ["natgen", "format"]:
            print(f"[nominal partial]\n{entry[2]['prompt']}")
            print(f"[perturbed partial]\n{entry[1]['prompt']}")
        else:
            print(f"[nominal]\n{entry[0]['prompt']}")
            print(f"[perturbed]\n{entry[1]['prompt']}")

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


def calculate_metric(perturbed_data_list, metric, nominal_data):
    length = len(nominal_data)
    if metric == "passatk":
        # perturbed pass@k
        passatk_list = []
        passatk_worst = {}
        for ndata in nominal_data:
             passatk_worst[ndata["task_id"]] = True
        for perturbed_data in perturbed_data_list:
            passatk_list.append(calculate_passatk(perturbed_data))
            for pdata in perturbed_data:
                assert pdata["task_id"] in passatk_worst
                passatk_worst[pdata["task_id"]] = passatk_worst[pdata["task_id"]] and pdata["passed"]
        worst_cnt = 0
        for key in passatk_worst:
            if passatk_worst[key]: 
                worst_cnt += 1
        return passatk_list, worst_cnt / length if passatk_list else " "

    if metric == "drop":
        # (nominal pass@k - perturbed pass@k) / nominal pass@k
        nominal_passatk = calculate_passatk(nominal_data)
        passatk_list = []
        passatk_worst = {}
        for ndata in nominal_data:
             passatk_worst[ndata["task_id"]] = True
        for perturbed_data in perturbed_data_list:
            perturbed_passatk = calculate_passatk(perturbed_data)
            passatk_list.append((nominal_passatk - perturbed_passatk) / nominal_passatk)
            for pdata in perturbed_data:
                assert pdata["task_id"] in passatk_worst
                passatk_worst[pdata["task_id"]] = passatk_worst[pdata["task_id"]] and pdata["passed"]
        worst_cnt = 0
        for key in passatk_worst:
            if passatk_worst[key]: 
                worst_cnt += 1
        perturbed_passatk_worst = worst_cnt / length
        return passatk_list, (nominal_passatk - perturbed_passatk_worst) / nominal_passatk if passatk_list else " "

    if metric == "relative":
        # (nominal != perturbed) / total prompts
        passatk_worst = {}
        for ndata in nominal_data:
            passatk_worst[ndata["task_id"]] = True
        nominal_dict = {}
        for ndata in nominal_data:
            nominal_dict[ndata["task_id"]] = ndata["passed"]
        relative_list = []
        for perturbed_data in perturbed_data_list:
            relative_cnt = 0
            for pdata in perturbed_data:
                assert pdata["task_id"] in passatk_worst
                passatk_worst[pdata["task_id"]] = passatk_worst[pdata["task_id"]] and pdata["passed"]
                if nominal_dict[pdata["task_id"]] != pdata["passed"]:
                    relative_cnt += 1
            relative_list.append(relative_cnt / length)
        worst_cnt = 0
        for key in passatk_worst:
            if nominal_dict[key] != passatk_worst[key]:
                worst_cnt += 1
        return relative_list, worst_cnt / length  if relative_list else " "

    if metric == "attack_success":
        # (nominal correct & perturbed incorrect) / nominal correct
        passatk_worst = {}
        for ndata in nominal_data:
            passatk_worst[ndata["task_id"]] = True
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
                assert pdata["task_id"] in passatk_worst
                passatk_worst[pdata["task_id"]] = passatk_worst[pdata["task_id"]] and pdata["passed"]
                if nominal_dict[pdata["task_id"]] and not pdata["passed"]:
                    success_cnt += 1
            success_list.append(success_cnt / correct_cnt)
        worst_cnt = 0
        for key in passatk_worst:
            if nominal_dict[key] and not passatk_worst[key]:
                worst_cnt += 1
        return success_list, worst_cnt / correct_cnt  if success_list else " "


def report_results(args):
    """ report all the nominal and perturbed completion results by the models
    >>> example1: python run_robust.py report nlaugmenter --models codegen-350M-mono codegen-350M-multi --datasets humaneval mbpp --n_outputs 5
    >>> example2: python run_robust.py report nlaugmenter --aug_method 0 --models codegen-350M-mono codegen-350M-multi --datasets humaneval mbpp --n_outputs 5 --metric drop
    >>> example3: python run_robust.py report natgen --models codegen-350M-mono codegen-350M-multi codegen-6B-mono codegen-6B-multi incoder-1B incoder-6B --datasets humaneval mbpp --metric passatk
    """
    results = {} # save all the results for saving csv table
    for dataset in args.datasets:
        results[dataset] = {}
        for model in args.models:
            results[dataset][model] = {}
    for model in args.models:
        for dataset in args.datasets:
            print(f"[{model}, {dataset}]")
            # baseline nominal completion path
            if args.method == "format": # partial code completion
                # data_nominal_completed_path = f"{model_generate_path}/{model}/full/{dataset}_partial/nominal/greedy/samples.jsonl_passatk.txt"
                data_nominal_completed_path = f"{model_generate_path}/{model}/full/{dataset}_partial/nominal/greedy/samples.jsonl_results.jsonl"
            elif args.method == "natgen": # black normalized partial code completion
                data_nominal_completed_path = f"{model_generate_path}/{model}/full/{dataset}/natgen/{dataset}_Black/s0/greedy/samples.jsonl_results.jsonl"
            else: # regular dataset completion
                data_nominal_completed_path = f"{model_generate_path}/{model}/full/{dataset}/nominal/greedy/samples.jsonl_results.jsonl"
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
                    data_perturbed_completed_path = f"{model_generate_path}/{model}/full/{dataset}/{args.method}/{dataset}_{RECIPES[args.method][aug_method]}/s{seed}/greedy/samples.jsonl_results.jsonl"
                    # print(data_perturbed_completed_path)
                    if os.path.exists(data_perturbed_completed_path):
                        perturbed_data_list.append(read_json(data_perturbed_completed_path))
                    else:
                        # print(f"{data_perturbed_completed_path} not exists, skip..")
                        pass
                passatk_list, passatk_worst = calculate_metric(perturbed_data_list, args.metric, data_nominal_completed)
                if passatk_list:
                    print(f"\t{RECIPES[args.method][aug_method]} {args.metric}: {passatk_list}, {passatk_worst:.4f}")
                else:
                    print(f"\t{RECIPES[args.method][aug_method]} {args.metric}: {passatk_list}, {passatk_worst}")
                # print(f"\t{RECIPES[args.method][aug_method]} {args.metric}: {passatk_worst:.4f}")
                results[dataset][model][RECIPES[args.method][aug_method]] = passatk_worst

    # reformulate results to csv table
    for dataset in args.datasets:
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


if __name__ == '__main__':
    """ The main function for using our robustness benchmark
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('status', type=str, choices=['perturb', 'create_partial', 'nominal', 'subset', 'exec', 'analysis', 'report'], help='The funcitons enabled by our benchmark')
    parser.add_argument('method', type=str, choices=["normal", "nlaugmenter", "natgen", "format", "func_name", "random"], help="The classes of perturbation. Please set method to natgen with status nominal to evaluate nominal partial code.")
    parser.add_argument('--aug_method', type=int, default=None, help="The detailed augmentation method used with index (index defined in config.json for each method). Default None means running all the perturbations")
    parser.add_argument('--datasets', nargs='+', default=["humaneval"], help='A list of datasets to perturb/evaluate with')
    parser.add_argument('--models', nargs='+', default=["codegen-350M-multi"], help="A list of the models needed to evaluate with (or create subset dataset for perturbed dataset, not needed most of the times).")
    parser.add_argument('--n_outputs', type=int, default=1, help="The total number of perturbations generated/evaluated with")
    parser.add_argument('--ngpus', type=int, default=1, help="The number of gpus to use.")
    parser.add_argument('--overwrite', action="store_true", help="Set overwrite to True if regenerate dataset perturbation or evaluation")
    parser.add_argument('--subset', type=str, default="full", choices=["full", "correct", "incorrect"], help="Using the whole dataset or only subsample for targeted models (deprecated).")
    parser.add_argument('--metric', type=str, default="passatk", choices=["passatk", "drop", "relative", "attack_success"], help="The metric used for reporting results.")
    parser.add_argument('--eval_only', action="store_true", help="Only want to reevaluate model generated completions.")
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
        report_results(args)



