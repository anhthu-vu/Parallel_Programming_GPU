# Parallel_Programming_GPU
This repository contains a mini project that I have done during the *Parallel Programming on GPU* course in M2A program at Sorbonne Université. In this mini project, we for Asian option pricing. To be more specific,
- In `MC.cu` file, we implemented the Nested Monte-Carlo method to simulate, using on [CUDA](https://developer.nvidia.com/cuda-toolkit) GPUs, the price of an Asian option $F$ for various starting values of $S_t$ and $I_t$ at any time $t$ in the grid $\left(\frac{1}{100}, \frac{2}{100}, ..., \frac{99}{100}, 1\right)$. This was done to collect a dataset of sample $(t, S_t, I_t, F)$, defined as follow:
  + $I_t = \int_0^t S_sds$
  + $dS_t = S_t \sigma dW_t$ where $W$ is a Brownian motion and $\sigma = 0.2$ is the volatility
  + $F(t, x, y) = E(X|S_t = x, I_t=y)$ where $X = (S_T-I_T)_+$
  + $S_0 = 100, T=1$

