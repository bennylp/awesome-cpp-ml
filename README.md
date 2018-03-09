# Awesome C++ BLAS/Matrix Libraries and Machine/Reinforcement Learning Frameworks

My curated list of C++ (GPU) matrix/BLAS libraries and machine learning/reinforcement learning frameworks.

## Free/Open Source BLAS/Matrix Libraries

Libraries:
- [ViennaCL](http://viennacl.sourceforge.net/): a free open-source linear algebra library for computations on many-core architectures (GPUs, MIC) and multi-core CPUs. The library is written in C++ and supports CUDA, OpenCL, and OpenMP (including switches at runtime).
- [CUSP](https://github.com/cusplibrary/cusplibrary): a library for sparse linear algebra and graph computations based on Thrust. Cusp provides a flexible, high-level interface for manipulating sparse matrices and solving sparse linear systems.
- [MAGMA](http://icl.cs.utk.edu/magma/): a dense linear algebra library similar to LAPACK but for heterogeneous/hybrid architectures, starting with current "Multicore+GPU" systems.
- [NVidia cuBLAS](https://developer.nvidia.com/cublas): a fast GPU-accelerated implementation of the standard basic linear algebra subroutines (BLAS).
- [CULA](http://www.culatools.com/): a set of GPU-accelerated linear algebra libraries utilizing the NVIDIA CUDA parallel computing architecture to dramatically improve the computation speed of sophisticated mathematics.
- [Boost uBlas](http://www.boost.org/doc/libs/1_59_0/libs/numeric/ublas/doc/): a C++ template class library that provides BLAS level 1, 2, 3 functionality for dense, packed and sparse matrices. The design and implementation unify mathematical notation via operator overloading and efficient code generation via expression templates.
- [CUV](https://github.com/deeplearningais/CUV): a C++ template and Python library which makes it easy to use NVIDIA(tm)
CUDA.
- [Eigen](https://eigen.tuxfamily.org): a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.
- [Armadillo](http://arma.sourceforge.net/): a high quality linear algebra library (matrix maths) for the C++ language, aiming towards a good balance between speed and ease of use.

The features listed here are based on my casual observations on Dec 2017. If you see a feature is not checked, it could be because it is not supported or I didn't find it/didn't have time to find it out.

Comparison:

|              | ViennaCL |   CUSP   |  MAGMA   |  cuBLAS  |   CULA   |   uBLAS  |   CUV    |  Eigen   |Armadillo |
|--------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| Language     |    C++   |    C++   |     C    |    C     |     C    |    C++   |   C++    |  c++98   | C++98 - 14 |
| License      | BSD like | Apache   | BSD like | free     | free/$   |  Boost   | BSD like |  MPL2    | Apache 2 |
| Created      | 2010     | 2009     |          | 2007?    |   2009   |   2004   |          |   2009   |   2008   |
| Last release | Jan 2016 | Apr 2015 | Nov 2017 |          | Apr 2014 | May 2016 | Sep 2015 | Jun 2017 | Dec 2017 |
| Active       |    ?     |    ✗     |    ✓     |    ✓     |    ✗     |     ?    |    ✗     |    ✓     |    ✓     |
|              |          |          |          |          |          |          |          |          |          |
|              | ViennaCL |   CUSP   |  MAGMA   |  cuBLAS  |   CULA   |   uBLAS  |   CUV    |  Eigen   |Armadillo |
| Platforms:   |          |          |          |          |          |          |          |          |          |
| - CPU        |    ✓     |          |    ✓     |          |          |     ✓    |    ✓     |    ✓     |    ✓     |
| - GPU        |    ✓     |    ✓     |    ✓     |    ✓     |    ✓     |          |    ✓     |    ✓     | partial  |
| - OpenCL     |    ✓     |          |    ✓     |          |          |          |          |          |          |
| - Xeon Phi   |    ✓     |          |    ✓     |          |          |          |          |          |          |
| - OpenMP     |    ✓     |          |    ✓     |          |          |          |          |          |    ✓     |
|              |          |          |          |          |          |          |          |          |          |
| Thrust compat|    ✓     |    ✓     |          |          |          |          |          |          |          |
|              |          |          |          |          |          |          |          |          |          |
|              | ViennaCL |   CUSP   |  MAGMA   |  cuBLAS  |   CULA   |   uBLAS  |   CUV    |  Eigen   |Armadillo |
| Features:    |          |          |          |          |          |          |          |          |          |
| - Column or row major| both | both |          |  column  |  column  |          |   both   |   both   |  column  |
| - Dense matrix|    ✓    |    ✓     |          |          |    ✓     |          |          |    ✓     |    ✓     |
| - Sparse matrix|   ✓    |    ✓     |          |          |    ✓     |          |          |    ✓     |    ✓     |
| - Slice/view |     ✓    |    ✓     |          |          |          |    ✓     |          |    ✓     |    ✓     |
| - BLAS L1    |    ✓     |    ✓     |    ✓     |    ✓     |    ✓     |    ✓     |    ✓     |    ✓     |    ✓     |
| - BLAS L2    |    ✓     |    ✓     |    ✓     |    ✓     |    ✓     |    ✓     |          |    ✓     |    ✓     |
| - BLAS L3    |    ✓     |    ✓     |    ✓     |    ✓     |    ✓     |    ✓     |          |    ✓     |    ✓     |
|              |          |          |          |          |          |          |          |          |          |
|              | ViennaCL |   CUSP   |  MAGMA   |  cuBLAS  |   CULA   |   uBLAS  |   CUV    |  Eigen   |Armadillo |
| Other:       |          |          |          |          |          |          |          |          |          |
| - fancy operators|  ✓   |    -     |          |          |          |          |          |          |    ✓     |
| - need Boost?| partly   |          |          |          |          |    ✓     |    ✓     |          |          |
|              |          |          |          |          |          |          |          |          |          |
| Notable users| Singa, <10 |        |          |   many   |          |          |          | TensorFlow, Shogun, 70+ | MLPACK, 30+ |
|              |ViennaCL|  CUSP   |  MAGMA   |  cuBLAS  |   CULA   |   uBLAS  |   CUV    |  Eigen   |Armadillo |


## Machine Learning Frameworks

Frameworks:
- [Darknet](https://pjreddie.com/darknet/): open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation. Mainly geared towards CNNs but has some RNNs as well.
- [MLPack](http://mlpack.org/): a scalable machine learning library, written in C++, that aims to provide fast, extensible implementations of cutting-edge machine learning algorithms.
- [Shogun](http://shogun-toolbox.org/): open-source machine learning library that offers a wide range of efficient and unified machine learning methods.
- [OpenNN](http://www.opennn.net/): an open source class library written in C++ programming language which implements neural networks with deep architectures, a main area of machine learning research.
- [DLib](http://dlib.net/ml.html): contains a wide range of machine learning algorithms. All designed to be highly modular, quick to execute, and simple to use via a clean and modern C++ API. See [paper (PDF)](http://jmlr.csail.mit.edu/papers/volume10/king09a/king09a.pdf).
- [Caffe](http://caffe.berkeleyvision.org/): a deep learning framework made with expression, speed, and modularity in mind. Caffe is geared towards CNN (Caffe stands for Convolutional Architecture for Fast Feature Embedding). See [paper (PDF)](https://arxiv.org/pdf/1408.5093.pdf).
- [Dynet](https://github.com/clab/dynet): neural network library by Carnegie Mellon University, aimed to work well with networks that have dynamic structures that change for every training instance. Seems to be geared towards sequence model (RNN/LSTM). See [paper (PDF)](https://arxiv.org/pdf/1701.03980.pdf).
- [Shark](http://image.diku.dk/shark/): Shark is a fast, modular, general open-source C++ machine learning library. Contains many basic ML algorithms comparable to scikit-lean such as linear regression, SVM, neural networks, clustering, etc. See [paper (PDF)](http://www.jmlr.org/papers/volume9/igel08a/igel08a.pdf).
- [Fido](http://fidoproject.github.io/): Fido is a light-weight, open-source, and highly modular C++ machine learning library. The library is targeted towards embedded electronics and robotics.

Comparison:

|              |Darknet |  MLPack  |  Shogun  |  OpenNN  | DLib |Caffe| Dynet | Shark |  Fido  |        |        |
|--------------|:------:|:--------:|:--------:|:--------:|:----:|:---:|:-----:|:-----:|:------:|:------:|:------:|
| License      |copyleft| BSD like |  GPLv3   |  LGPLv3  | Boost| BSD |Apache |  LGPL |   MIT  |        |        |
| Created      |  2013  |   2011   |   1999   |   2012   | 2006 | 2013| 2015  |  2008 |  2015? |        |        |
| Active       |    ~   |    ✓     |    ✓     |    ✓     |  ✓   |  ✓  |   ✓   |   ✓   |    ~   |        |        |
|              |        |          |          |          |      |     |       |       |        |        |        |
| Platforms:   |        |          |          |          |      |     |       |       |        |        |        |
| - GPU        |    ✓   |    -     |    ?     |    ✓     |  ✓   |  ✓  |   ✓   |       |        |        |        |
| - OpenMP     |    ✓   |    -     |    ?     |    ✓     |      |  ✓  |       |       |        |        |        |
| - OpenCL     |        |          |          |          |      |  ~  |       |       |        |        |        |
| - Windows    |        |          |    ✓     |    ✓     |  ✓   |  ~  |   ✓   |   ✓   |        |        |        |
|              |        |          |          |          |      |     |       |       |        |        |        |
| Features:    |        |          |          |          |      |     |       |       |        |        |        |
| - Supervised |    -   |    ✓     |    ✓     |    ✓     |  ✓   |  ✓  |   ✓   |   ✓   |    ~   |        |        |
| - Unsupervised|   -   |          |    ✓     |    -     |  ✓   |     |       |   ✓   |        |        |        |
| - RL         |    -   |    ✓     |    -     |    -     |  ~   |  ✓  |       |       |    ✓   |        |        |
| - CNN        |    ✓   |          |          |          |  ✓   |  ✓  |       |       |        |        |        |
| - RNN        |    ✓   |          |          |          |      |     |   ✓   |       |        |        |        |
|              |        |          |          |          |      |     |       |       |        |        |        |
| Matrix lib   |  own   | Armadillo|          |  Eigen   | own  | own | Eigen | uBLAS | -/STL  |        |        |
|              |        |          |          |          |      |     |       |       |        |        |        |
| Notable users|        |          |          |          |  10+ |1000+|  10+  |       |        |        |        |


Others:
- [frugally-deep](https://github.com/Dobiasd/frugally-deep): header-only library for using Keras models in C++, supporting CNN. Currently only runs on CPU.

## Reinforcement Learning

- [RLlib](https://github.com/HerveFrezza-Buet/RLlib) - C++ library for reinforcement learning. See the paper "[A C++ Template-Based Reinforcement Learning Library: Fitting the Code to the Mathematics](http://www.jmlr.org/papers/v14/frezza-buet13a.html)", Hervé Frezza-Buet, Matthieu Geist.
- [RLLib (Saminda's)](http://web.cs.miami.edu/home/saminda/rllib.html) - a lightweight C++ template library that implements  incremental, standard, and gradient temporal-difference learning algorithms in Reinforcement Learning. See the [paper](http://robocup.oss-cn-beijing.aliyuncs.com/symposium%2FRoboCup_Symposium_2015_submission_3.pdf) (PDF).

## Other Lists

- [Awesome Machine Learning (C++ Section)](https://github.com/josephmisiti/awesome-machine-learning#cpp)
- [http://mloss.org/software/](http://mloss.org/software/) - machine learning OSS.
- [Comparison of deep learning software](https://en.wikipedia.org/wiki/Comparison_of_deep_learning_software) - Wikipedia

## Other Stuff

### Machine Learning for Trading

Software:
- [Q Learning for Trading](https://github.com/ucaiado/QLearning_Trading): an adaptive learning model to trade a single stock under the reinforcement learning framework. 

Papers:
- [Optimal Asset Allocation using Adaptive Dynamic Programming](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.48.2563). Ralph Neuneier. 1996
- [Artificial Markets and Intelligent Agents](http://cbcl.mit.edu/cbcl/publications/theses/thesis-chan.pdf). N.T. Chan. 2001
- [Learning to Trade via Direct Reinforcement](https://bi.snu.ac.kr/SEMINAR/Joint2k1/ojm5.pdf). John Moody and Matthew Saffell, IEEE Transactions on Neural Networks, Vol 12, No 4, July 2001
- [Reinforcement learning for trading systems and portfolios](https://vvvvw.aaai.org/Papers/KDD/1998/KDD98-049.pdf). John Moody , Matthew Saffell. 1998
- [Algorithm Trading using Q-Learning and Recurrent Reinforcement Learning](http://cs229.stanford.edu/proj2009/LvDuZhai.pdf). Du, Xin, Jinjian Zhai, and Koupin Lv. 2009
- [The price impact of order book events. Journal of financial econometrics](https://pdfs.semanticscholar.org/d064/5eb3d744f9e962ff09b8a5e9156f2147e983.pdf). R. Cont, k. Arseniy, and S. Sasha. 2014
- [Agent Inspired Trading Using Recurrent Reinforcement Learning and LSTM Neural Networks](https://arxiv.org/abs/1707.07338). David W. Lu. 2017.
