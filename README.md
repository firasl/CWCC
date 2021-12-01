# Description
This repository provides the official implimentation of the illuminant estimation algorithm **CWCC** proposed in paper *[Robust channel-wise illumination estimation](https://arxiv.org/abs/2111.05681)* accepted in *[BMVC2021](https://www.bmvc2021.com/)* using *[INTEL-TAU dataset](https://ieeexplore.ieee.org/document/9371681)*. 


# CWCC
Recently, Convolutional Neural Networks (CNNs) have been widely used to solve the illuminant estimation problem and have often led to state-of-the-art results. Standard approaches operate directly on the input image. In this paper, we argue that this problem can be decomposed into three channel-wise independent and symmetric sub-problems and propose a novel CNN-based illumination estimation approach based on this decomposition. The proposed method substantially reduces the number of parameters needed to solve the task while achieving competitive experimental results compared to state-of-the-art methods. Furthermore, the practical application of illumination estimation techniques typically requires identifying the extreme error cases. This can be achieved using an uncertainty estimation technique. In this work, we propose a novel color constancy uncertainty estimation approach that augments the trained model with an auxiliary branch which learns to predict the error based on the feature representation. Intuitively, the model learns which feature combinations are robust and are thus likely to yield low errors and which combinations result in erroneous estimates. We test this approach on the proposed method and show that it can indeed be used to avoid several extreme error cases and, thus, improves the practicality of the proposed technique.


**Motivation** 

Formally, RGB values of an image <img src="https://render.githubusercontent.com/render/math?math=\textbf{I}"> at every pixel <img src="https://render.githubusercontent.com/render/math?math=(x,y)">  are expressed as a function of the global illuminant <img src="https://render.githubusercontent.com/render/math?math=\textbf{e}">, the original colors $\textbf{R}(x,y)$ as follows:

<img src="https://render.githubusercontent.com/render/math?math=\textbf{I}(x,y) = \textbf{R}(x,y) \circ \textbf{e}">

where <img src="https://render.githubusercontent.com/render/math?math=\circ"> is element-wise multiplication. Illumination estimation refers to the problem of estimating <img src="https://render.githubusercontent.com/render/math?math=\textbf{e}"> given an input <img src="https://render.githubusercontent.com/render/math?math=\textbf{I}">. Most CNN-based illuminant estimation approaches operate directly on the input image <img src="https://render.githubusercontent.com/render/math?math=\textbf{I}"> without exploiting the specificities and characteristics the aforementioned equation defining the problem. In fact, it is easy to see that the illumination estimation problem can be divided into three problems using the color channels (r,g,b):

<img src="https://render.githubusercontent.com/render/math?math=\textbf{I}_r =  \textbf{R}_r   \textbf{e}_r">

<img src="https://render.githubusercontent.com/render/math?math=\textbf{I}_g =  \textbf{R}_g   \textbf{e}_g">

<img src="https://render.githubusercontent.com/render/math?math=\textbf{I}_b =  \textbf{R}_b   \textbf{e}_b">

We note that the sub-equations in this decomposition are linear and symmetric, i.e., the problem defined in each equation is similar. We propose a novel CNN-based illuminant estimation approach, called **CWCC**, which leverages the decomposition enabling us to reduce the number of parameters up to 90\%.

**Model** 
**CWCC** is composed of two blocks, the disjoint block and the merging block. The disjoint block learns to solve each sub-equation separately. To this end, each color channel has a separate CNN sub-model. Moreover, we exploit the symmetry of the sub-problems by sharing the weights of  'filter blocks' of the three sub-models.  In the merging block,  we concatenate the outputs of each channel of the first block. Then, we use a model which acts on this mixed representation and aims to learn the optimal way to merge the feature maps of each channel and approximate the illuminant $\textbf{e}$.
<p align="center">
  <img src="figures/intro22.jpg" width="600">
</p>



