#  Towards Greener Yet Powerful Code Generation via Quantization:An Empirical Study

This is the repo for paper [Towards Greener Yet Powerful Code Generation via Quantization:An Empirical Study](https://arxiv.org/pdf/2303.05378.pdf), to be published in FSE 2023. In this paper, we extensively study the impact of quantized model on code generation tasks across different dimension: (i) resource usage and carbon footprint, (ii) accuracy, and (iii) robustness. To this end, through systematic experiments we find a recipe of quantization technique that could run even a 6B model in a regular laptop without significant accuracy or robustness degradation.

## Installation
We are using python 3.8, cuda 11.6. Anaconda would be recommended. Please run the following commands for installation.
```
conda deactivate; conda env remove --name code_quant
conda create --name code_quant python=3.8
conda activate code_quant
```

Installing huggingface for model inference
```
pip install transformers==4.21.1
pip install -U torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```


## License
The released code is under Apache-2.0 license.


## Cite this work

Please cite with the following bibtex

```
@inproceedings{quant2023,
  title = {Towards Greener Yet Powerful Code Generation via Quantization:An Empirical Study},
  author = {Xiaokai Wei,
  Sujan Kumar Gonugondla,
  Shiqi Wang,
  Wasi Ahmad,
  Baishakhi Ray,
  Haifeng Qian,
  Xiaopeng Li,
  Varun Kumar,
  Zijian Wang,
  Yuchen Tian,
  Qing Sun,
  Ben Athiwaratkun,
  Mingyue Shang,
  Murali Krishna Ramanathan,
  Parminder Bhatia,
  Bing Xiang},
  booktitle={Proceedings of the 31th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering},
  year = {2023},
}

```
