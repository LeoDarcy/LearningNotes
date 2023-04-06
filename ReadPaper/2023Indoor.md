# Indoor / NeRF

## Learning Object-Compositional Neural Radiance Field for Editable Scene Rendering  (ICCV 2021)

问题：建模的NeRF无法编辑

解决方法：把背景和物体分开来建模。用二维的物体分割作为监督。标记的k个物体就用k个物体的field来学习。
第二个是考虑到遮挡的问题。如果直接用物体分割的mask来监督物体的训练，那么会学到有个shattered破碎的结果。我们利用场景分支的透射率来指导目标分支的偏采样，我们称之为scene guidance。本质上就是减少遮蔽部分的采样和监督。（需要综合代码来看）

数据集 scannet 和toydesk

## BakedSDF: Meshing Neural SDFs for Real-Time View Synthesis （2023.2 arxiv Google Research）

目标：用Neural SDF生成mesh

## Make-It-3D: High-Fidelity 3D Creation from A Single Image with Diffusion Prior （2023.3 arxiv Microsoft&上交）

Two-stage framework

第一阶段：先用NeRF基于一张图片重建出来。重建对象先用mask勾选出来，NeRF只对这个mask以内的color进行优化。创新点：使用上diffusion prior。具体方法：使用一个image captioning model 将图片变成text，然后再计算score distillation sampling loss $\mathcal{L}_{\text{SDS}}$

这个loss是基于latent space来做的。

遇到问题：虽然这个loss能衡量图片和text的相似度，但是我们还需要align到输入的图片中，才能生成一致的图片。因此使用一个diffusion CLIP loss。也就是将reference图片和渲染出来的图片通过一个CLIP的encoder，通过loss使得这两者的latent一致。

但是需要注意的是我们并不是同时使用这两个loss的，我们在小的timestamp上使用CLIP的loss，在大的timestamp上使用SDS。实验证明使用两者之后能够生成合适的细节。

另外一个创新点在于depth prior使用。前面的图片虽然能够生成比较好的图片，但是物理上还是会shape ambiguity。细节：先使用一个单目深度估计来对输入图片进行深度估计。

## Look Outside the Room: Synthesizing A Consistent Long-Term 3D Scene Video from A Single Image （CVPR 2022）

待解决的问题：给定一些相机轨迹和单张图像，需要生成其他相机视点的图片形成视频。这是一个生成问题。

解决思路：目标是生成一个连续的长时间轴的视频，因此提出使用一个transformer来序列建模这个有locality constraints的连续图像。按照一帧一帧来做，给定前面帧后用模型来估计下一帧的图像。但是学习这样的sequential model其实很难。解决这个问题的key observation是：不是每个relational pair都是同等重要的，只需要关注重点的部分。直觉上，相邻两帧可以找到共同的部分和新生成的部分。因此提出Camera-aware Bias。在使用self-attention operation的时候用一个MLP估计bias加入到affinity matrix上。（这样做的好处是？）

实现细节：输入是一个sequence，输出是新的一帧。

细节实现上：分成两个阶段：第一阶段使用VQ-GAN（Vector Quantized）将图片变成tokens。tokens在codebook里面找最相似的来量化成$z$

根据$z$就可以在codebook里面找到对应的向量，继而可以decode出来。

对于camera encoder，假设相机是pinhole camera，就有内参外参

带来两个问题：一是随着序列长度增加，self-attention并不能很好地建模两个patch之间的关系。二是long term的关系中需要考虑更多设计。

问题一用camera-aware bias解决

dataset: Matterport3D， RealEstate10K

## NeuralRoom: Geometry-Constrained Neural Implicit Surfaces for Indoor Scene Reconstruction (2022 ToG)
