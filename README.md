# Decentralized Gaussian Processess for Multi-Robot Systems

Demonstration code of decentralized methods for Gaussian processes [1]. For further details check the paper [here](http://www.georgekontoudis.com/). A 3-minute presentation of the paper can be found [here](https://youtu.be/kVnQ0uNy-sY).

## Contents

The code implements:
* Decentralized Nested Pointwise Aggregation of Experts (DEC-NPAE)
* Distributed Nested Pointwise Aggregation of Experts (DIST-NPAE)

The source code of the factorized training and the centralized NPAE [2] can be found in the GRBCM [3] [GitHub repository](https://github.com/LiuHaiTao01/GRBCM).

## Execution

Install the [gpml toolbox](http://www.gaussianprocess.org/gpml/code/matlab/doc/).

Execute:
```
demo_2D.m
```

## References

[1] G. P. Kontoudis and D. J. Stilwell, “Decentralized Nested Gaussian Processes for Multi-Robot Systems,” in *IEEE International Conference on Robotics and Automation (ICRA)*, 2021.

[2] D. Rullière, N. Durrande, F. Bachoc, and C. Chevalier, “Nested Kriging predictions for datasets with a large number of observations,” *Statistics and Computing*, 2018.

[3] H. Liu, J. Cai, Y. Wang, and Y. S. Ong, “Generalized robust Bayesian committee machine for large-scale Gaussian process regression,” in *International Conference on Machine Learning (ICML)*, 2018.

## Notes

Please open a [GitHub issue](https://github.com/gkontoudis/decentralized-GP/issues) if you encounter any problem or send me an email at gpkont@vt.edu.
