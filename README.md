# Parallel_Programming_GPU
This repository contains a mini project that I have done during the *Parallel Programming on GPU* course in M2A program at Sorbonne Université. In this mini project, we for Asian option pricing. To be more specific,

- In `MC.cu` file, we implemented Nested Monte-Carlo method to simulate on GPUs using [CUDA](https://developer.nvidia.com/cuda-toolkit) the price of an Asian option for many choices of starting values of $S_t$ and $I_t$ at any time $t$ to collect a dataset of sample $(t, S_t, I_t, F)$ defined as below:
