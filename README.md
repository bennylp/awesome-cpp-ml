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

|              | [ViennaCL](http://viennacl.sourceforge.net/) | [CUSP](https://github.com/cusplibrary/cusplibrary) | [MAGMA](http://icl.cs.utk.edu/magma/) | [cuBLAS](https://developer.nvidia.com/cublas) | [CULA](http://www.culatools.com/) | [uBlas](http://www.boost.org/doc/libs/1_59_0/libs/numeric/ublas/doc/) | [CUV](https://github.com/deeplearningais/CUV) | [Eigen](https://eigen.tuxfamily.org) | [Armadillo](http://arma.sourceforge.net/) |
|--------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| Language     |    C++   |    C++   |     C    |    C     |          |          |          |  c++98   | C++98-14 |
| License      | BSD like | Apache   | BSD like | free     | free/$   |  Boost   | BSD like |  MPL2    | Apache 2 |
| Created      | 2010     | 2009     |          | 2007?    |   2009   |          |          |   2009   |          |
| Last release | Jan 2016 | Apr 2015 | Nov 2017 |   ✓      | Apr 2014 |          | Sep 2015 | Jun 2017 |          |
| Last commit  | Aug 2017 |          |          |          |          |          |          |          |          |
|              |          |          |          |          |          |          |          |          |          |
| Platforms:   |          |          |          |          |          |          |          |          |          |
| - CPU        |    ✓     |          |    ✓     |          |          |     ✓    |    ✓     |    ✓     |    ✓     |
| - GPU        |    ✓     |    ✓     |    ✓     |    ✓     |    ✓     |          |    ✓     |    ✓     | partial  |
| - OpenCL     |    ✓     |          |    ✓     |          |          |          |          |          |          |
| - Xeon Phi   |    ✓     |          |    ✓     |          |          |          |          |          |          |
| - OpenMP     |    ✓     |          |    ✓     |          |          |          |          |          |    ✓     |
|              |          |          |          |          |          |          |          |          |          |
| Thrust compat|    ✓     |    ✓     |          |          |          |          |          |          |          |
|              |          |          |          |          |          |          |          |          |          |
| Features:    |          |          |          |          |          |          |          |          |          |
| - Column or row major| both | both |          |  column  |          |          |   both   |          |  column  |
| - Dense matrix|    ✓    |    ✓     |          |          |          |          |          |          |    ✓     |
| - Sparse matrix|   ✓    |    ✓     |          |          |          |          |          |          |    ✓     |
| - Slice/view |     ✓    |    ✓     |          |          |          |    ✓     |          |          |    ✓     |
| - BLAS L1    |    ✓     |    ✓     |    ✓     |    ✓     |    ✓     |    ✓     |    ✓     |    ✓     |    ✓     |
| - BLAS L2    |    ✓     |    ✓     |    ✓     |    ✓     |    ✓     |    ✓     |          |    ✓     |    ✓     |
| - BLAS L3    |    ✓     |    ✓     |    ✓     |    ✓     |    ✓     |    ✓     |          |    ✓     |    ✓     |
|              |          |          |          |          |          |          |          |          |          |
| Other:       |          |          |          |          |          |          |          |          |          |
| - fancy operators|  ✓   |    -     |          |          |          |          |          |          |    ✓     |
| - need Boost?| partly   |          |          |          |          |    ✓     |    ✓     |          |          |
|              |          |          |          |          |          |          |          |          |          |
| Notable users:| Singa, <10 |       |          |   many   |          |          |          | TensorFlow, Shogun, 70+ | MLPACK, 30+ |
| =============== |ViennaCL|  CUSP   |  MAGMA   |  cuBLAS  |   CULA   |   uBLAS  |   CUV    |  Eigen   |Armadillo |

