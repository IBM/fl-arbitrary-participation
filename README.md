### Federated Learning with Arbitrary Client Participation

This is the official code for the paper:
- Shiqiang Wang, Mingyue Ji, "[A Unified Analysis of Federated Learning with Arbitrary Client Participation](https://arxiv.org/abs/2205.13648)," in NeurIPS 2022.
    ```
    @inproceedings{wang2022unified,
     author = {Wang, Shiqiang and Ji, Mingyue},
     booktitle = {Advances in Neural Information Processing Systems},
     pages = {19124-19137},
     title = {A Unified Analysis of Federated Learning with Arbitrary Client Participation},
     volume = {35},
     year = {2022}
    }
    ```

The code was run successfully in the following environment: Python 3.8, PyTorch 1.7, Torchvision 0.8.1

All configurations can be found in the `config.py` file.

The following are commands to obtain the results in Figure 1 of the main paper. The CSV filename indicates the method used with each command.


FashionMNIST:
```
python3 main.py -data fashion -availability periodic -seeds 1,2,3,4,5,6,7,8,9,10 -lr-warmup 0.1 -iters-warmup 10000 -iters-total 1500000 -lr 0.1 -lr-global 1.0 -wait-all 1 -full-batch 0 -out fashion_wait_minibatch.csv

python3 main.py -data fashion -availability periodic -seeds 1,2,3,4,5,6,7,8,9,10 -lr-warmup 0.1 -iters-warmup 10000 -iters-total 1500000 -lr 0.1 -lr-global 1.0 -wait-all 1 -full-batch 1 -out fashion_wait_full.csv

python3 main.py -data fashion -availability periodic -seeds 1,2,3,4,5,6,7,8,9,10 -lr-warmup 0.1 -iters-warmup 10000 -iters-total 1500000 -lr 0.00001 -lr-global 1.0 -out fashion_alg1_no_amplify.csv

python3 main.py -data fashion -availability periodic -seeds 1,2,3,4,5,6,7,8,9,10 -lr-warmup 0.1 -iters-warmup 10000 -iters-total 1500000 -lr 0.00001 -lr-global 10.0 -out fashion_alg1_amplify.csv
```

CIFAR-10:
```
python3 main.py -data cifar -availability periodic -seeds 1,2,3,4,5 -lr-warmup 0.05 -iters-warmup 20000 -iters-total 3000000 -lr 0.05 -lr-global 1.0 -wait-all 1 -full-batch 0 -out cifar_wait_minibatch.csv 

python3 main.py -data cifar -availability periodic -seeds 1,2,3,4,5 -lr-warmup 0.05 -iters-warmup 20000 -iters-total 3000000 -lr 0.05 -lr-global 1.0 -wait-all 1 -full-batch 1 -out cifar_wait_full.csv 

python3 main.py -data cifar -availability periodic -seeds 1,2,3,4,5 -lr-warmup 0.05 -iters-warmup 20000 -iters-total 3000000 -lr 0.00005 -lr-global 1.0 -out cifar_alg1_no_amplify.csv 

python3 main.py -data cifar -availability periodic -seeds 1,2,3,4,5 -lr-warmup 0.05 -iters-warmup 20000 -iters-total 3000000 -lr 0.000005 -lr-global 10.0 -out cifar_alg1_amplify.csv 
```

See Section D.1 in the appendix of the paper for additional explanation including the learning rate choices.

The results are saved in the file specified by the `-out` argument, in CSV format.

Note: The results from the code show the number of iterations. They need to be converted to the number of rounds to obtain results in the paper.

This code was inspired by and derived from past work with other collaborators, such as:
- https://github.com/PengchaoHan/EasyFL/
- https://github.com/IBM/adaptive-federated-learning
