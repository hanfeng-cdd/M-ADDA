# M-ADDA: Metric-based Adversarial Discriminative Domain Adaptation [[Paper]](https://arxiv.org/abs/1807.02552)

## Requirements

- Pytorch version 0.4 or higher.

## Running pretrained models

To obtain the test results, run the following command,

```
python main.py -e usps2mnist mnist2usps uspsBig2mnistBig mnistBig2uspsBig -m test_model
```

The output should be,

```
mnist2usps          0.955676
mnistBig2uspsBig    0.980541
usps2mnist          0.951500
uspsBig2mnistBig    0.983100
```
which represent the accuracies obtained on the target test set.

- mnistBig, and uspsBig use the full training set.
- mnist, and usps use 2000 images from MNIST and 1800 images from USPS for training, respectively.

## Training the models

To train the source and target models run the command,

```
python main.py -e usps2mnist mnist2usps uspsBig2mnistBig mnistBig2uspsBig -m train -rt 1 -rs 1
```
Source (MNIST)            |  Target (USPS)
:-------------------------:|:-------------------------:
![](figures/src_mnistBig2uspsBig.png)  |  ![](figures/tgt_mnistBig2uspsBig.png)

## Citation 
If you find the code useful for your research, please cite:

```bibtex
@Article{laradji2018m,
    title={M-ADDA: Unsupervised Domain Adaptation with Deep Metric Learning},
    author={Laradji, Issam and Babanezhad, Reza},
    journal = {arXiv},
    year = {2018}
}
```
