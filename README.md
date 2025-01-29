# Decentralized Gaussian Processess for Multi-Agent Systems

Demonstration code of decentralized Gaussian process training [1,2] and prediction [3]. For further details check the paper for GP training [here](https://www.georgekontoudis.com/publications/Access24_Kontoudis_DecentralizedGPTraining.pdf) and prediction [here](https://www.georgekontoudis.com/publications/ICRA21_Kontoudis_DistributedNestedGaussianProcesses.pdf). A presentation of the GP training paper can be found [here](https://youtu.be/8Tz8ande5Gk?si=6xbKSXk0W6og94Ww) and prediction [here](https://youtu.be/kVnQ0uNy-sY).

## Contents

The code implements:
* Centralized Generalized Analytical Proximal GP Training (gapx-GP) [1]
* Decentralized Consensus GP Training (DEC-c-GP) [1]
* Decentralized Analytical Proximal GP Training (DEC-apx-GP) [2]
* Decentralized Generalized Analytical Proximal GP Training (DEC-gapx-GP) [1]
* Decentralized Nested Pointwise Aggregation of Experts (DEC-NPAE) [3]
* Distributed Nested Pointwise Aggregation of Experts (DIST-NPAE) [3]

The source code of the factorized training and the centralized NPAE [4] can be found in the GRBCM [5] [GitHub repository](https://github.com/LiuHaiTao01/GRBCM).

## Execution

Install the [gpml toolbox](http://www.gaussianprocess.org/gpml/code/matlab/doc/).

Execute:
```
demo_2D.m
```

## References

[1] G. P. Kontoudis and D. J. Stilwell, “Scalable, Federated Gaussian Process Training for Decentralized Multi-Agent Systems,” in *IEEE Access*, 2024.

[2] G. P. Kontoudis and D. J. Stilwell, “Decentralized Federated Learning using Gaussian Processes,” in *IEEE International Symposium on Multi-Robot and Multi-Agent Systems (MRS)*, 2023.

[3] G. P. Kontoudis and D. J. Stilwell, “Decentralized Nested Gaussian Processes for Multi-Robot Systems,” in *IEEE International Conference on Robotics and Automation (ICRA)*, 2021.

[4] D. Rullière, N. Durrande, F. Bachoc, and C. Chevalier, “Nested Kriging predictions for datasets with a large number of observations,” *Statistics and Computing*, 2018.

[5] H. Liu, J. Cai, Y. Wang, and Y. S. Ong, “Generalized robust Bayesian committee machine for large-scale Gaussian process regression,” in *International Conference on Machine Learning (ICML)*, 2018.

## Notes

Please open a [GitHub issue](https://github.com/gkontoudis/decentralized-GP/issues) if you encounter any problem or send me an email at gpkont@vt.edu.
