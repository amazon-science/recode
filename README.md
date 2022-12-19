# ReCode: Robustness Evaluation of Code Generation Models

This is the repo for ReCode, providing a comprehensive evaluation for the practical robustness of code generation models like CodeGEN, Incoder, GPT-J, CodeT5. In specific, this benchmark provides over 30 different general perturbations on docstrings, function names, and codes. The perturbations are carefully selected and implemented such that the perturbed datasets are naturally and semantically close to the original non-perturbed datasets. All the perturbations are well implemented with automatic generation, providing easy usage and customization. With these perturbations available in our benchmark, the user can get to know a comprehensive analysis of model robustness performance.

Our benchmark is general with regard to datasets and models. Given the perturbed datasets, the users can evaluate any of public/customized code generation models with the default inference provided by our benchmark. We also allow users to provide their own inference scripts to evaluate robustness in our benchmark by replacing `evaluate-public-models/run_eval_models.sh`.

After the model evaluation is done on perturbed datasets, we provide overall robustness analysis for the evaluated models such that the users can easily compare across different models and get aware of the possible practical robustness problems.

Lastly, we release a standard version of the perturbed datasets for HumanEval and MBPP in this benchmark for general robustness evaluation and compare across different models proposed in future works.

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

Installing humaneval. Need to enable humaneval by uncommenting out execution line in `execution.py`.
```
cd evaluate-public-models
# git clone https://github.com/openai/human-eval # we already provide the humaneval files in repo
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

## Running our benchmarks
We have four main types of perturbations. Multiple variances are defined and implemented for each type of perturbation. One can find detailed config in `config.py`. To run our models, please config the data and model path correctly in `config.py` and then run the following command to create partial code and perturb datasets with each type of perturbations.
```
python run_robust.py create_partial natgen # preparing partial code for code perturbations
python run_robust.py perturb nlaugmenter # perturb with nlaugmenter
python run_robust.py perturb func_name # perturb with function rename
python run_robust.py perturb natgen # perturb with code structure transformation
python run_robust.py perturb code # perturb with code format transformation
```

One can specify augmentation method for each type of perturbations with --aug_method, index can be found in config.py. --datasets allow to specify perturbed datasets.
```
python run_robust.py perturb func_name --aug_method 0 --datasets humaneval mbpp # perturb with function rename CamelCase (index=0 defined in config.py) on humaneval and mbpp
``` 

To evaluate the models, one can run:
```
python run_robust.py nominal normal # nominal evaluation with non-perturbed datasets
python run_robust.py nominal natgen # nominal evaluation with non-perturbed partial code datasets
python run_robust.py exec nlaugmenter # nlaugmenter perturbed datasets evaluation
python run_robust.py exec func_name # function rename perturbed datasets evaluation
python run_robust.py exec natgen # code structure perturbed datasets evaluation
python run_robust.py exec format # code format transformation perturbed datasets evaluation
```

If one wants to perturb or evaluate specific augmentation method, one can easily run
```
python run_robust.py perturb func_name --aug_method 0 # perturb dataset with function rename CamelCase (index=0 defined in config.py) 
python run_robust.py exec func_name --aug_method 0 # evaluate model on dataset with function rename CamelCase (index=0 defined in config.py) 
```

For targeted models please use augments --models and --datasets. Note that one has to correctly config the model and dataset path and names correctly.
```
python run_robust.py perturb func_name --datasets humaneval mbpp --models codegen-350M-multi codegen-350M-mono # perturb dataset humaneval mbpp on codegen-350M-multi and codegen-350M-mono
python run_robust.py exec func_name --datasets humaneval mbpp --models codegen-350M-multi codegen-350M-mono # evaluate model on dataset humaneval mbpp on codegen-350M-multi and codegen-350M-mono
```

To analyze the evaluated results, one can run the following commands. Report option summarizes the results while analysis option provide prints for the perturbed data and completion by the model.
```
python run_robust.py report func_name --models codegen-350M-multi --datasets mbpp # get results for dataset perturbed with function rename by codegen-350M-multi
python run_robust.py analysis func_name --models codegen-350M-multi --datasets mbpp # analyze completion samples for dataset perturbed with function rename by codegen-350M-multi
```

To debug and customize perturbations, one can use low-level APIs. Turn on --print_sample to debug and check customized perturbations on each sample.
```
python perturb.py --method format --aug_method 0
```


## License
The ReCode benchmark is under Apache-2.0 license.


## Authors
This code generation robustness benchmark (ReCode) is developed by a team in Amazon AWS

- Shiqi Wang, wshiqi@amazon.com, (main developer)
- Zheng Li, zl634@cornell.edu
- Haifeng Qian, qianhf@amazon.com
- Mingyue Shang, myshang@amazon.com
- Chenghao Yang, ychengha@amazon.com
- Zijian Wang, zijwan@amazon.com
- Varun Kumar, kuvrun@amazon.com
- Samson Tan, samson@amazon.com
- Baishakhi Ray, rabaisha@amazon.com
- Parminder Bhatia, parmib@amazon.com
- Murali Krishna Ramanathan, mkraman@amazon.com
- Bing Xiang, bxiang@amazon.com