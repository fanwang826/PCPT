# implementation for **PTDP** in zero-shot test-time adaptation
 

### Prerequisites:
- python == 3.7.3
- pytorch == 1.0.1.post2
- torchvision == 0.2.2
- single-GPU configuration
- numpy, scipy, sklearn, PIL, argparse, tqdm


### Dataset:

- For fine-grained cross-dataset generalization, we adopt the same train/val/test splits as CoOp. Please refer to [this page](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md#how-to-install-datasets), and look for download links of datsets and `split_zhou_${dataset_name}.json`, and put the json files under `./data/data_splits/`. 

- Full results and details of source model test-time adaptation and test-time generaliztaion will be released.


### zerp-shot test-time adaptation:

* Using a hand-crafted prompt as initialization (*e.g.,* "a photo of a ___")
* Using a <ins> learned soft prompt</ins> ([CoOp](https://arxiv.org/abs/2109.01134)) as initialization.
	
##### PTDP on the fine-grained classification dataset
	
- Adaptation to target domains in zero-shoy test-time adaptation with PTDP, Pseudo-labeling and TPT

    ```
    python PTDP_zero_shot.py --alpha 1.0 --beta 0.1 --test_sets Flower102
    ```
    ```
    python Pseudo-labeling_zero_shot.py --test_sets Flower102 
    ```
   