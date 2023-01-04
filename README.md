# ReCode: Robustness Evaluation of Code Generation Models

This is the repo for ReCode ([arXiv](https://arxiv.org/abs/2212.10264)), providing a comprehensive evaluation for the practical robustness of code generation models like CodeGEN, Incoder, GPT-J. In specific, this benchmark provides over 30 different general perturbations on docstrings, function names, and codes. The perturbations are carefully selected and implemented such that the perturbed datasets are naturally and semantically close to the original non-perturbed datasets. All the perturbations are well implemented with automatic generation, providing easy usage and customization. With these perturbations available in our benchmark, the user can get to know a comprehensive analysis of model robustness performance.

Our benchmark is general with regard to datasets and models. Given the perturbed datasets, the users can evaluate any of public/customized code generation models with the default inference provided by our benchmark. We also allow users to provide their own datasets and models to evaluate robustness in our benchmark by configuring `config.json` and inference script `evaluate-public-models/run_eval_models.sh`.

After the model evaluation is done on perturbed datasets, we provide overall robustness analysis for the evaluated models such that the users can easily compare across different models and get aware of the possible practical robustness problems.

Lastly, we release a standard version of the perturbed datasets `dataset-release/perturbed-finalized` for HumanEval and MBPP in this benchmark for general robustness evaluation and compare across different models proposed in future works.

## Installation
We are using python 3.8, cuda 11.6. Anaconda would be recommended. Please run the following commands for installation.
```
conda deactivate; conda env remove --name ReCode
conda create --name ReCode python=3.8
conda activate ReCode
```

Installing huggingface for model inference
```
pip install transformers==4.21.1
pip install -U torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

Installing humaneval. Need to enable humaneval by uncommenting out execution line `exec(check_program, exec_globals)` in `execution.py`.
```
cd evaluate-public-models
git clone https://github.com/openai/human-eval
pip install -e human-eval
cd ..
```

Installing nlaugmenter for perturbations
```
cd nlaugmenter
pip install -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz
cd ..
```

Installing treesitter for perturbations. Note that we customized our code syntax perturbatons based on [natgen](https://github.com/saikat107/NatGen). 
```
cd natgen/treesitter
git clone https://github.com/tree-sitter/tree-sitter-python # clone the py-tree-sitter
python build.py # build my-languages.so file
cd ../transformations
ln -s ../treesitter/build/my-languages.so ./
pip install sympy
cd ../..
```

## Running our ReCode benchmark
We provide general APIs for running our benchmark in `run_robust.py`. We have four main types of perturbations (1) nlaugmenter on docstrings (nlaugmenter) (2) function rename (func_name) (3) code syntax (natgen) (4) code format (format). Multiple variances are defined and implemented for each type of perturbation. One can find detailed config in `config.json`. 

Overall we have multiple steps for benchmark as described in detail in the following sections: (1) [perturb] creating perturbed datasets, (2) [exec] run the models on nominal/perturbed datasets, and (3) [report_coarse] collect and summarize running results according to our proposed robustness metrics.

### Step1: Create perturbed datasets [perturb] 

[perturb] option is used to create perturbed datasets. One can run the following commands to perturb based on one's own nominal datasets (path config in `config.json`). 

Note that we also released our perturbed data used for evaluation in paper as a general robustness benchmark (`dataset-release/perturbed_finalized`). To directly evaluate on our created benchmark datasets, please change `output_adv_path` in `config.json` to that path and skip all the following commands for perturbing in this [perturb] section!


```
python run_robust.py create_partial natgen # preparing partial code for code perturbations
python run_robust.py perturb nlaugmenter # perturb with nlaugmenter transformations on docstrings
python run_robust.py perturb func_name # perturb with function rename
python run_robust.py perturb natgen # perturb with code syntax transformations
python run_robust.py perturb code # perturb with code format transformations
```

One can specify augmentation method for each type of perturbations with --aug_method, index can be found in `config.json`. --datasets allow to specify perturbed datasets.
```
python run_robust.py perturb func_name --aug_method 0 --datasets humaneval mbpp # perturb with function rename CamelCase (index=0 defined in config.json) on humaneval and mbpp
``` 

Our benchmark also provides [analysis] option to check how each sample is perturbed one by one
```
python run_robust.py analysis func_name --aug_method 0 --models None # check perturbed data one by one with function rename CamelCase (index=0 defined in config.json)
```

To debug and customize perturbations, one can use low-level APIs. Turn on --print_sample to debug and check customized perturbations on each sample.
```
python perturb.py --method format --aug_method 0 --print_sample
```

### Step2: Run on perturbed datasets [exec] 

[exec] option is used for evaluating targeted models on perturbed datasets. To evaluate models with our benchmark, please config the targeted nominal/perturbed datasets and model path correctly in `config.json`. One can then run with:
```
python run_robust.py nominal normal # nominal evaluation with non-perturbed datasets
python run_robust.py nominal natgen # nominal evaluation with non-perturbed partial code datasets
python run_robust.py exec nlaugmenter # nlaugmenter perturbed datasets evaluation
python run_robust.py exec func_name # function rename perturbed datasets evaluation
python run_robust.py exec natgen # code structure perturbed datasets evaluation
python run_robust.py exec format # code format transformation perturbed datasets evaluation
```

If one wants to evaluate specific augmentation method, one can easily run
```
python run_robust.py exec func_name --aug_method 0 # evaluate model on dataset with function rename CamelCase (index=0 defined in config.json)
```

For targeted models please use augments --models and --datasets. Note that one has to correctly config the model names and path correctly in the running shell file in `run_script` in `config.json`. Detailed running hyperparameters can be configured in that shell file. Please make sure that shell file can run correctly for nominal evaluation on your own models/datasets. Our benchmark will mainly call that file for evaluation. The default one is `evaluate-public-models/run_eval_models.sh`
```
python run_robust.py perturb func_name --datasets humaneval mbpp --models codegen-350M-multi codegen-350M-mono # perturb dataset humaneval mbpp on codegen-350M-multi and codegen-350M-mono
python run_robust.py exec func_name --datasets humaneval mbpp --models codegen-350M-multi codegen-350M-mono # evaluate model on dataset humaneval mbpp on codegen-350M-multi and codegen-350M-mono
```

### Step3: Summarize running results [report_coarse]

In our paper, we proposed three main robustness metrics: robust pass@k, robust drop@k, and robust relative@k. To summarize and collect the evaluated results, one can run the following commands. In specific, `report_coarse` option summarizes the robustness numbers for all thee metrics (as shown in main tables in paper). `report` option summarizes the detailed robustness results into csv (detailed tables in appendix of paper). The results will be saved as tables in `csv_coarse` and `csv`.
```
python run_robust.py report_coarse func_name --models codegen-350M-multi codegen-350M-mono --datasets humaneval # get summarized results for dataset perturbed with function rename
python run_robust.py report func_name --models codegen-350M-multi codegen-350M-mono --datasets humaneval # get detailed results for dataset perturbed with function rename
```

`analysis` option with --models given provides prints for the perturbed data and completion by the model for each prompt.
```
python run_robust.py analysis func_name --models codegen-350M-mono --datasets humaneval # analyze completion samples for dataset perturbed with function rename by codegen-350M-mono
```


## License
The ReCode benchmark is under Apache-2.0 license.


## Cite this work

Please cite with the following bibtex

```
@article{recode_wang2022,
  title = {ReCode: Robustness Evaluation of Code Generation Models},
  author = {Wang, Shiqi and
   Zheng, Li and
   Qian, Haifeng and
   Yang, Chenghao and
   Wang, Zijian and
   Kumar, Varun and
   Shang, Mingyue and
   Tan, Samson and
   Ray, Baishakhi and
   Bhatia, Parminder and
   Nallapati, Ramesh and
   Ramanathan, Murali Krishna and
   Roth, Dan and
   Xiang, Bing},
  doi = {10.48550/arXiv.2212.10264},
  url = {https://arxiv.org/abs/2212.10264},
  keywords = {Machine Learning (cs.LG), Computation and Language (cs.CL)},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}

```
