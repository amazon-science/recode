""" This is the file to perturb targeted dataset
Please run python perturb.py --help to see detailed usage.
A recommended high-level API is to call python run_robust.py perturb <method> ...
"""

import json
import argparse
import torch
import random
import os
from tqdm import tqdm
import json
from nlaugmenter import *
from natgen import *
from format import *
from func_rename import *


def set_env():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def set_seed(seed, deterministic=True):
    """ Set up random seeds
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        # torch.use_deterministic_algorithms(deterministic)


def get_languages(data):
    if data in ["humaneval", "mbpp"]:
        return "python"
    if data in ["python"]:
        return data
    print(f"language for {data} not supported")
    exit()

def load_data(args, task, data_file=None, data_path=None):
    """ loading data
    if any data_file provided, directly load it
    if data_file is None, data_path has to provide for clean data loading
    return loaded data
    """
    if data_file is None:
        # this is the clean dataset
        assert data_path is not None, "no data_path provided"
        if task == "code_generation" or args.create_partial_code:
            data_file = os.path.join(data_path, DATASET_PATH[args.data])
        elif task == "partial_code":
            data_file = os.path.join(data_path, args.data + "_partial.jsonl")
            if not os.path.exists(data_file): 
                print("please enable --create_partial_code to create partial dataset first!")
                exit()
        else:
            print(f"task {task} not supported!")

    data = []
    with open(data_file, 'r') as input_file:
        for line in input_file:
            data.append(json.loads(line))

    print(f"loading data in {data_file} with {len(data)} prompts")
    return data


def apply_subset_filter(args, data):
    """ Apply subset filter for targeted models (deprecated)
    """
    nominal_file_path = f"nominal_results.json"
    if not os.path.exists(nominal_file_path):
        print("please run nominal collect first")
        exit()
    nominal_file = open(nominal_file_path, "r")
    nominal_results = json.load(nominal_file)[args.data]
    nominal_file.close()

    filter_data = []
    diff = 0
    for idx in range(len(data)):
        prompt = data[idx]
        task_id = prompt["task_id"]
        if args.subset == "correct":
            print(task_id, nominal_results[task_id])
            if nominal_results[task_id] == 1:
                filter_data.append(data[idx])
                if not "perturbed" in data[idx]:
                    print("Warning: no perturbed property saved!")
                if "perturbed" in data[idx] and data[idx]["perturbed"]:
                    diff += 1
        elif args.subset == "incorrect":
            if nominal_results[task_id] == 0:
                filter_data.append(data[idx])
                if "perturbed" in data[idx] and data[idx]["perturbed"]:
                    diff += 1
        else:
            print("only support subset filter as correct/incorrect!")
            exit()

    print(f"[{args.subset}] subset filter {len(filter_data)} prompts, {diff} were perturbed")
    return filter_data


def write_generated_data(args, output_adv_path, generated_data):
    """ Write the generated data to output_adv_path
    """
    with open(output_adv_path, "wb") as output_file:
        for output_line in generated_data:
            output_file.write((json.dumps(output_line) + "\n").encode('utf-8'))
    print(f"{len(generated_data)} generated prompts saved. Done...\n")


def ptb_doc(doc, ptb, seed=0, black_list=[]):
    """ Using nlaugmenter to perturb the dataset
    """
    # not using max_outputs since some nl transformation does not have this argument
    t = ptb(seed=seed)#(max_outputs = n_outputs)
    # t = ptb(prob=0.2)
    new_doc, replaces = preprocess_docstring(black_list, doc, t)
    try:
        new_doc = t.generate(new_doc)[0]
        new_doc = postprocess_docstring(new_doc, replaces)
        return new_doc
    except:
        print("Warning: error during transformation!")
        return doc


def ptb_entry(args, entry, ptb, seed=0):
    head, doc, cases = sep_doc(args.data, entry['prompt'], entry["entry_point"])
    # we need to maintain a blacklist for variable names, function names, and type names such that we will not perturb these names
    if args.data in ["humaneval", "mbpp"]: 
        code_string = entry["prompt"] + entry["canonical_solution"]
    elif args.data in ["mbjp", "mbjsp"]: 
        code_string = entry["prompt"]
    else:
        print(f"data {args.data} not supported for docstring perturbation")
    black_list = word_blacklist(code_string, args.data)
    ptbd_doc = ptb_doc(doc, ptb, seed, black_list)
    res = {k:v for k, v in entry.items()}
    res['prompt'] = ''.join([head, ptbd_doc, cases])
    res['seed'] = seed
    return res


def perturb_nlaug(args, data, recipes):
    """ The function to perturb docstring with NL_AUG_RECIPES
    """
    ptb_type = eval(recipes[args.aug_method])
    diff = 0
    generated_data = []
    for idx, entry in tqdm(enumerate(data)):
        # if idx != 10: continue
        if args.print_sample: print(f"=== origin [{entry['task_id']}] ===\n", entry['prompt'])
        output_line = ptb_entry(args, entry, ptb_type, args.seed)
        if args.print_sample: print(f"=== perturbed with {recipes[args.aug_method]}, seed {args.seed} ===\n", output_line['prompt'])
        # save perturbed property
        output_line["perturbed"] = False
        if output_line["prompt"] != entry["prompt"]:
            output_line["perturbed"] = True
            diff += 1
        generated_data.append(output_line)
        if args.print_sample: import pdb; pdb.set_trace()
    print(f"{ptb_type.name()} has {diff} perturbations")
    return generated_data


def create_partial_code(data):
    """ The function to create partial code
    """
    generated_data = []
    for idx, entry in tqdm(enumerate(data)):
        res = {}
        for k, v in entry.items():
            res[k] = v

        whole = res['prompt'] + res['canonical_solution']
        header, doc, body = sep(whole, res['entry_point'])
        if args.print_sample:
            print(" === orig === ")
            print(f"header:\n{header}")
            print(f"doc:\n{doc}")
            print(f"body:\n{body}")
        
        num_lines = count_lines(body)
        assert num_lines > 0
        if num_lines == 1:
            res['partial'] = None
        else:
            # find the middle line to split
            # half = (num_lines + 1) // 2
            half = (num_lines) // 2 # make nominal partial a little harder
            tmp = 0
            while half > 0:
                idx = body.find('\n', tmp)
                tmp = idx + 1
                half -= 1
            res['prompt'] = header + doc + body[:idx + 1]
            res['canonical_solution'] = body[idx + 1:]

            post_indent_buffer = "" # get the indent for the line after this split line
            for ch in body[idx + 1:]:
                if ch in [" ", "\t"]:
                    post_indent_buffer += ch
                else:
                    break
            pre_indent_buffer = "" # get the indent for the line ahead this split line
            for ch in body[body[: idx].rfind('\n') + 1: idx]:
                if ch in [" ", "\t"]:
                    pre_indent_buffer += ch
                else:
                    break
            
            # add additional comment line for print can avoid black to detect wrong format, need to be fixed for natgen
            # have to be very careful about the indent since we want to have code executable
            indent_buffer = post_indent_buffer if len(pre_indent_buffer) < len(post_indent_buffer) else pre_indent_buffer
            split_print_flag = indent_buffer + "# print('@@this is the line to split##')\n"
            res['partial'] = res['prompt'] + split_print_flag + res['canonical_solution']

        if args.print_sample:
            print(" === partial split === ")
            print(res['partial'])
            import pdb; pdb.set_trace()
        generated_data.append(res)
    return generated_data


def perturb_partial(args, data, recipes):
    """ The function to perturb partial code with PARTIAL_RECIPES
    """
    generated_data = []
    diff = 0
    for idx, entry in tqdm(enumerate(data)):
        # if entry["task_id"] != "MBPP/11": continue
        res = {}
        for k, v in entry.items():
            res[k] = v
        
        if "partial" not in res: import pdb; pdb.set_trace()
        if res["partial"] is not None:
            header, doc, body = sep(res["partial"], res['entry_point'])
        else:
            # only 1 line return code in canonical solution
            header, doc, body = sep(res['prompt'] + res['canonical_solution'], res['entry_point'])
        
        code = header + body
        indent_type = detect_indent_type(res["prompt"], res['entry_point'])

        if args.print_sample:
            print(f" === orig [{res['task_id']}] === ")
            print(res["prompt"])
            print("================")

        new_doc = doc
        if recipes[args.aug_method] == "Black" or res["partial"] is None:
            # Only black normalization or no partial code (just one line return)
            new_code = code
        else:
            new_code = code
            # new_code = res["prompt"]
            if args.data in ["humaneval", "mbpp"]:
                new_code = new_code.replace(";\n", "\n").replace(";", "\n")
            if "@@this is the line to split##" in code:
                # uncomment the split such that deadcode can also be inserted before this line; If comment, indent will be wrong with deadcode insertion
                new_code = new_code.replace("# print(\'@@this is the line to split##\')", "print(\'@@this is the line to split##\')")
            # tsf = eval(PARTIAL_RECIPES[args.aug_method])("languages.so", "python")
            tsf = eval(recipes[args.aug_method])("natgen/languages.so", get_languages(args.data))
            # first half = True for prompt based perturbations
            new_code, meta = tsf.transform_code(code=new_code, first_half=True)
            new_code = beautify_python_code(new_code.split()).replace("\\", "")
            # make doc indent to be \t to match natgen format
            new_doc = black_tablize_doc(doc, indent_type)
            if "@@this is the line to split##" in code:
                # uncomment the split such that it will not be removed by natgen
                new_code = new_code.replace("print(\'@@this is the line to split##\')", "# print(\'@@this is the line to split##\')")
            # print(new_code)
            # import pdb; pdb.set_trace()

        # add doc into the transformed new code
        new_header, _, new_body = sep(new_code, res['entry_point'])
        new_code = new_header + new_doc + new_body
        # use black to do the normalization
        new_code, resp = black_reformat(new_code, orig_code=res)
        # make sure this sep works correctly for mbpp
        # new_header, new_doc, new_body = sep(new_code)
        
        if res["partial"] is not None and resp:
            # new_code = new_header + new_doc + new_body
            # idx = new_code.find("@@this is the line to split##")
            idx = new_code.find("# print(\'@@this is the line to split##\')")
            line_end = idx - 1
            while new_code[idx] in [" ", "\t"]:
                line_end -= 1
            res["prompt"] = new_code[: line_end]
            if res["prompt"][-1] != "\n":
                res["prompt"] += "\n"
            res["canonical_solution"] = new_code[line_end + 1 :]
            if res["prompt"] == "":
                res["prompt"] = header + new_doc
                res["canonical_solution"] = new_body
        else:
            res["prompt"] = entry["prompt"]
            res["canonical_solution"] = entry["canonical_solution"]

        if args.print_sample:
            print(f" === perturbed with {recipes[args.aug_method]} === ")
            print(res["prompt"])
            print("================")
            import pdb; pdb.set_trace()
        
        if res["prompt"] != entry["prompt"]: diff += 1
        generated_data.append(res)

    print(f"{recipes[args.aug_method]} has {diff} perturbations")
    return generated_data


def perturb_format(args, data, recipes):
    """ The function to perturb with FORMAT_RECIPES
    """
    generated_data = []
    for entry in data:
        res = {}
        for key in entry:
            res[key] = entry[key]
        if args.print_sample:
            print(" === orig === ")
            print(entry["prompt"])
            print("================")

        res["prompt"] = eval(recipes[args.aug_method])(entry["prompt"], entry["entry_point"], language=get_languages(args.data))

        if args.print_sample:
            print(f" === perturbed with {recipes[args.aug_method]} === ")
            print(res["prompt"])
            print("================")
            import pdb; pdb.set_trace()
        
        generated_data.append(res)
    return generated_data


def perturb_func_name(args, data, recipes):
    """ The function to perturb with function renaming in FUNC_RECIPES
    """
    generated_data = []
    for entry in data:
        res = {}
        for key in entry:
            res[key] = entry[key]
        if args.print_sample:
            print(" === orig === ")
            print(entry["prompt"])

        res["prompt"], res["entry_point"] = eval(recipes[args.aug_method])(entry["prompt"], entry["entry_point"], seed=args.seed)

        if args.print_sample:
            print(f" === perturbed with {recipes[args.aug_method]} (success: {res['prompt']!=entry['prompt']}) === ")
            print(res["prompt"])
            import pdb; pdb.set_trace()
        
        generated_data.append(res)
    return generated_data


def perturb_random(args, data, recipes):
    """ The function to perturb with random perturbations in FULL_RECIPES
    """
    generated_data = []
    data_normal = load_data(args, "code_generation", data_file=None, data_path=data_path)
    data_partial = load_data(args, "partial_code", data_file=None, data_path=data_path)
    for pidx in range(len(data_normal)):
        coin = random.randint(0, len(recipes)-1)
        aug_method = FULL_RECIPES[coin]
        if aug_method in NL_AUG_RECIPES:
            perturb_func = perturb_nlaug
            args.aug_method = recipes.index(aug_method)
            entry = data_normal[pidx]
        elif aug_method in PARTIAL_RECIPES:
            perturb_func = perturb_partial
            args.aug_method = recipes.index(aug_method)
            entry = data_partial[pidx]
        elif aug_method in FORMAT_RECIPES:
            perturb_func = perturb_format
            args.aug_method = FORMAT_RECIPES.index(aug_method)
            entry = data_partial[pidx]
        elif aug_method in FUNC_RECIPES:
            perturb_func = perturb_func_name
            args.aug_method = FUNC_RECIPES.index(aug_method)
            entry = data_normal[pidx]
        else:
            exit()
        # print(pidx, aug_method, perturb_func, len(data_normal), len(data_partial))
        # print(entry["prompt"])
        res = perturb_func(args, [entry])[0]
        res["aug_method"] = aug_method
        generated_data.append(res)
    return generated_data


def read_config(config, data):
    with open(config, "r") as config_file:
        config_dict = json.load(config_file)
    NL_AUG_RECIPES = config_dict[data]["NL_AUG_RECIPES"]
    PARTIAL_RECIPES = config_dict[data]["PARTIAL_RECIPES"]
    FUNC_RECIPES = config_dict[data]["FUNC_RECIPES"]
    FORMAT_RECIPES = config_dict[data]["FORMAT_RECIPES"]
    FULL_RECIPES = NL_AUG_RECIPES + PARTIAL_RECIPES + FUNC_RECIPES + FORMAT_RECIPES
    RECIPES = config_dict["RECIPES"]
    for recipe in RECIPES:
        RECIPES[recipe] = eval(RECIPES[recipe])
    DATASET_PATH = config_dict["DATASET_PATH"]
    RANDOM_TRANS = config_dict["RANDOM_TRANS"]
    data_path = config_dict["data_path"]
    output_adv_path = config_dict["output_adv_path"]
    model_generate_path = config_dict["model_generate_path"]
    run_script = config_dict["run_script"]
    return NL_AUG_RECIPES, PARTIAL_RECIPES, FUNC_RECIPES, FORMAT_RECIPES, FULL_RECIPES, RECIPES,\
             DATASET_PATH, RANDOM_TRANS, data_path, output_adv_path, model_generate_path, run_script


if __name__ == '__main__':
    """ The main function to perform perturbations
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="humaneval", choices=["humaneval", "mbpp", "mbjp", "mbjsp", "mbkp", "mbphp", "mbrbp"])
    parser.add_argument('--config', default="config.json", help="The config to run.")
    parser.add_argument('--rng-seed', type=int, default=42, help="global random seed.")
    parser.add_argument('--rng-deterministic', type=bool, default=True)
    parser.add_argument('--task', type=str, default="code_generation", choices=["code_generation", "partial_code", "function_completion"], help="The task for generating perturbation.")
    parser.add_argument('--subset', type=str, default="full", choices=["full", "correct", "incorrect"], help="Using the whole dataset or only subsample for targeted models")
    parser.add_argument('--method', type=str, default="normal", choices=["normal", "nlaugmenter", "natgen", "format", "func_name", "random"], help="The classes of perturbation.")
    parser.add_argument('--aug_method', type=int, default=0, help="The detailed perturbation method used with index.")
    parser.add_argument('--overwrite', action="store_true", help="Whether to overwrite previously perturbed datasets.")
    parser.add_argument('--seed', type=int, default=0, help="random seed used for transformation.")
    parser.add_argument('--model', type=str, default="672M_old", help="The model needed to load nominal results and get subset filter (only needed when --subset is not full).")
    parser.add_argument("--output_name", type=str, default=None, help="The complete output jsonl by model")
    parser.add_argument("--create_partial_code", default=False, action='store_true', help="Set true if want to generate partial code dataset.")
    parser.add_argument('--print_sample', action="store_true", help="For debug purpose, print each perturbed sample with pdb stop.")
    args = parser.parse_args()

    NL_AUG_RECIPES, PARTIAL_RECIPES, FUNC_RECIPES, FORMAT_RECIPES, FULL_RECIPES, RECIPES, \
        DATASET_PATH, RANDOM_TRANS, data_path, output_adv_path, model_generate_path, run_script = read_config(args.config, args.data)

    set_env()
    set_seed(args.rng_seed + args.seed, deterministic=args.rng_deterministic)

    # config path for dataset
    output_adv_path = os.path.join(output_adv_path, args.data)
    if not os.path.exists(output_adv_path):
        os.system(f"mkdir {output_adv_path}")
    # config path for subset (full dataset, targeted model, correct/not correct)
    output_adv_path = os.path.join(output_adv_path, args.subset)
    if not os.path.exists(output_adv_path):
        os.system(f"mkdir {output_adv_path}")
    # config path for method
    output_adv_path = os.path.join(output_adv_path, args.method)
    if not os.path.exists(output_adv_path):
        os.system(f"mkdir {output_adv_path}")
    # config output file name
    if args.output_name is None:
        if args.method == "normal": 
            args.output_name = args.data
        elif args.method == "nlaugmenter":
            args.output_name = args.data + "_" + NL_AUG_RECIPES[args.aug_method]
        elif args.method == "natgen":
            args.output_name = args.data + "_" + PARTIAL_RECIPES[args.aug_method]
        elif args.method == "format":
            args.output_name = args.data + "_" + FORMAT_RECIPES[args.aug_method]
        elif args.method == "func_name":
            args.output_name = args.data + "_" + FUNC_RECIPES[args.aug_method]
        elif args.method == "random":
            # random has no aug_method option
            args.output_name = args.data + "_" + "random"
        else:
            print(f"method {args.method} not supported!")
            exit()
        # if args.seed != 0:
        args.output_name += f"_s{args.seed}"
    if args.output_name[:-6] != ".jsonl":
        args.output_name += ".jsonl"
    output_adv_path = os.path.join(output_adv_path, args.output_name)
    print(f"generated outputs will be saved in {output_adv_path}")
    # handle overwrite if exists
    if os.path.exists(output_adv_path) and not args.print_sample:
        print(f"{output_adv_path} exists")
        if not args.overwrite:
            print(f"Not overwrite, stop generating!")
            exit()
        print(f"removing {output_adv_path}...")
        os.system(f"rm {output_adv_path}")

    if args.method != "normal": assert args.aug_method < len(RECIPES[args.method])

    if args.create_partial_code:
        # we have to recreate partial code for regular code-generation dataset
        data = load_data(args, args.task, data_file=None, data_path=data_path)
        generated_data = create_partial_code(data)
        output_adv_path = os.path.join(data_path, args.data + "_partial.jsonl")
        print(f"generating partial code! redirect to {output_adv_path}")
        if os.path.exists(output_adv_path):
            print(f"{output_adv_path} exists")
            if not args.overwrite:
                print(f"Not overwrite, stop generating!")
                exit()
            print(f"recreating {output_adv_path}...")
            os.system(f"rm {output_adv_path}")
        write_generated_data(args, output_adv_path, generated_data)
        exit()
    

    if args.subset != "full":
        full_adv_path = output_adv_path.replace(args.subset, "full")
        assert os.path.exists(full_adv_path), "please run subset full to generate all data first!"
        # generated data already exists in full_adv_path
        # just apply subset filter, no need to regenerate the data
        print(f"full path exists: {full_adv_path}")
        print("creating subset data from full path")
        data = load_data(args, args.task, data_file=full_adv_path, data_path=None)
        # subset filter, filter generated data from full_adv_path according to the filter
        generated_data = apply_subset_filter(args, data)
    else: 
        # generating data in this branch
        # load clean data
        data = load_data(args, args.task, data_file=None, data_path=data_path)
        # do something
        if args.method == "normal":
            generated_data = data
        elif args.method == "nlaugmenter":
            generated_data = perturb_nlaug(args, data, NL_AUG_RECIPES)
        elif args.method == "natgen":
            assert args.task == "partial_code", "please add --tast partial_code for code syntax (natgen) perturbation"
            generated_data = perturb_partial(args, data, PARTIAL_RECIPES)
        elif args.method == "format":
            assert args.task == "partial_code", "please add --tast partial_code for code format (format) perturbation"
            generated_data = perturb_format(args, data, FORMAT_RECIPES)
        elif args.method == "func_name":
            generated_data = perturb_func_name(args, data, FUNC_RECIPES)
        elif args.method == "random":
            generated_data = perturb_random(args, data_path, FULL_RECIPES)

    if not args.print_sample:
        write_generated_data(args, output_adv_path, generated_data)

    




    
    
    
