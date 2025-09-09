# DiffRL_Finance

## Plan
1. build on simple diffusion model, make it work for time series data (yfinance) ✅
2. implement evaluation function to compare performance ✅
3. implement result visualization tool ✅
- 3.5 annotate / make a note of the code to fully understand what the hell is going on 
4. implement wavelet transformation and cascaded frequency decomposition as an input
5. implement subtle details from TimeGrad (specifically for processign time series data)
    - make input multivariate
    - update result visualization tool
6. implement conditioning for denoising process
7. implement KAN as a conditioning layer
8. implement a function that allows to see what is going on in KAN
    - Visualization tool possibly?

## Architecture Design
Three-tier hybrid architecture: 
1. Wavelet decomposition → Multi-scale frequency analysis 
2. WaveNet with FastKANConv residual blocks → Temporal modeling with learnable frequency-specific patterns 
3. KAN conditioning → Macroeconomic awareness


## Reference
https://github.com/huangst21/TimeKAN/blob/main/models/TimeKAN.py
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/zalandoresearch/pytorch-ts/blob/master/pts/model/time_grad/time_grad_network.py
https://huggingface.co/blog/annotated-diffusion
https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=LQnlc27k7Aiw