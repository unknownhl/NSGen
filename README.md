## NSGen
The official implementation of "Neuron Semantic Guided Test Generation for Deep Neural Networks Fuzzing"

## Implementations

This repo implements the **NSGen** proposed in our paper and previous neuron coverage criteria (optimized if possible), including

- [x] Neuron Coverage (**NC**) [1]
- [x] K-Multisection Neuron Coverage (**KMNC**) [2]
- [x] Neuron Boundary Coverage (**NBC**) [2]
- [x] Strong Neuron Activation Coverage (**SNAC**) [2]
- [x] Top-K Neuron Coverage (**TKNC**) [2]
- [x] Top-K Neuron Patterns (**TKNP**) [2]
- [x] Cluster-based Coverage (**CC**) [3]
- [x] Likelihood Surprise Coverage (**LSC**) [4,6]
- [x] Distance-ratio Surprise Coverage (**DSC**) [5,6]
- [x] Mahalanobis Distance Surprise Coverage (**MDSC**) [5,6]
- [x] Causal-Aware Coverage (**CAC**) [7] 
- [x] NeuraL Coverage (**NLC**) [8]

Each criterion is implemented as one Python class in `coverage.py`.

[1] *DeepXplore: Automated whitebox testing of deep learning systems*, SOSP 2017.  
[2] *DeepGauge: Comprehensive and multi granularity testing criteria for gauging the robustness of deep learning systems*, ASE 2018.  
[3] *Tensorfuzz: Debugging neural networks with coverage-guided fuzzing*, ICML 2019.  
[4] *Guiding deep learning system testing using surprise adequacy*, ICSE 2019.  
[5] *Reducing dnn labelling cost using surprise adequacy: An industrial case study for autonomous driving*, FSE Industry Track 2020.  
[6] *Evaluating Surprise Adequacy for Deep Learning System Testing*, TOSEM 2023.  
[7] *CC: Causality-Aware Coverage Criterion for Deep Neural Networks*, ICSE 2023.  
[8] *Revisiting Neuron Coverage for DNN Testing: A Layer-Wise and Distribution-Aware Criterion*, ICSE 2023.

## Model & Dataset
-The training code: please see [CIFAR10](https://github.com/kuangliu/pytorch-cifar) and [ImageNet](https://pytorch.org/vision/stable/models.html).  
-Dataset: please see [Dataset](https://drive.google.com/file/d/1vuLGrdorRihpEerTxnBg2SyZOdJG9EFp/view?usp=drive_link).

## Guiding Input Mutation in DNN Testing

```bash
python fuzz.py --dataset CIFAR10 --model resnet50 --criterion LG --ab_exp clip --device cuda:0
```
- `--model` - The tested DNN.  
chocies = [`resnet50`, `vgg16_bn`, `mobilenet_v2`]

- `--dataset` - Training dataset of the tested DNN. Test suites are generated using test split of this dataset.  
choices = [`CIFAR10`, `ImageNet`, `CIFAR100`, `Flowers102`]

- `--criterion` - The used criterion.  
choices = [`NC`, `KMNC`, `NBC`, `SNAC`, `TKNC`, `TKNP`, `CC`, `LSC`, `DSC`, `MDSC`, `CAC`, `NLC`, `LG`]


