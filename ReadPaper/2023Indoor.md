# Indoor / NeRF

## Learning Object-Compositional Neural Radiance Field for Editable Scene Rendering  (ICCV 2021)

问题：建模的NeRF无法编辑

解决方法：把背景和物体分开来只针对物体进行建模。用二维的物体分割作为监督。标记的k个物体就用k个物体的field来学习。
第二个是考虑到遮挡的问题。如果直接用物体分割的mask来监督物体的训练，那么会学到有个shattered破碎的结果。我们利用场景分支的透射率来指导目标分支的偏采样，我们称之为scene guidance。本质上就是减少遮蔽部分的采样和监督。（需要综合代码来看）

数据集 scannet 和toydesk

## BakedSDF: Meshing Neural SDFs for Real-Time View Synthesis （2023.2 arxiv Google Research）

目标：用Neural SDF生成mesh

abstract: 我们首先优化了一种混合神经体-表面场景表示，该场景表示设计为具有与场景中的表面对应的良好表现的水平集。然后，我们将这种表示bake成一个高质量的三角形网格，我们配备了一个基于球面高斯的简单而快速的依赖于视图的外观模型。最后对这样的baked的表示方式进行优化，使得可以利用加速多边形rasterization pipeline来在商用硬件上进行实时视角合成。

第一步：首先用一个SDF来获得density。实现时候结合啦mipnerf360和VolSDF的优点。SDF一个优先是可以定义表面的法向量，因此用上了Ref-NeRF的appearance model，吧appearance划分成diffuse和specular部分，反射和normal有关。（这部分需要细看refnerf的做法和真正实现）

## Make-It-3D: High-Fidelity 3D Creation from A Single Image with Diffusion Prior （2023.3 arxiv Microsoft&上交 开源）

代码[仓库](https://github1s.com/junshutang/Make-It-3D)

Two-stage framework

第一阶段：先用NeRF基于一张图片重建出来。重建对象先用mask勾选出来，NeRF只对这个mask以内的color进行优化。创新点：使用上diffusion prior。具体方法：使用一个image captioning model 将图片变成text，然后再计算score distillation sampling loss $\mathcal{L}_{\text{SDS}}$

这个loss是基于latent space来做的。

遇到问题：虽然这个loss能衡量图片和text的相似度，但是我们还需要align到输入的图片中，才能生成一致的图片。因此使用一个diffusion CLIP loss。也就是将reference图片和渲染出来的图片通过一个CLIP的encoder，通过loss使得这两者的latent一致。

但是需要注意的是我们并不是同时使用这两个loss的，我们在小的timestamp上使用CLIP的loss，在大的timestamp上使用SDS。实验证明使用两者之后能够生成合适的细节。

另外一个创新点在于depth prior使用。前面的图片虽然能够生成比较好的图片，但是物理上还是会shape ambiguity。细节：先使用一个单目深度估计来对输入图片进行深度估计。然后用估计的深度和NeRF自己产生的深度计算的negative Pearson correlation 来作为loss来监督。

## Layout-Guided Novel View Synthesis From a Single Indoor Panorama（2021 ICCV ）

给定单张全景图，估计另一个视角下的图像。metric使用的是新视角的SSIM PSNR

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

## Generative Novel View Synthesis with 3D-Aware Diffusion Models（2023 arxiv）

在上面的LOTR（Look outside the room）基础上做的更好。在AIGC的notes里面有介绍。使用的数据集：单个物体用的是shapeNet和CO3D，对于室内场景用的是matterport3D

## Neural 3D Scene Reconstruction with the Manhattan-world Assumption （CVPR 2022 oral）

使用的是7个scannet中的数据集，已经下载好了



## NeuralRoom: Geometry-Constrained Neural Implicit Surfaces for Indoor Scene Reconstruction (2022 ToG)

使用的是8个从scannet中选取的场景。

## PhyIR: Physics-based Inverse Rendering for Panoramic Indoor Images CVPR 2022

## TexIR:Multi-view Inverse Rendering for Large-scale Real-world Indoor Scenes CVPR 2023

数据集 ScanNet Matterport Replica

## Learning to Reconstruct 3D Non-Cuboid Room Layout from a Single RGB Image （2022WACV）

单图预测房间布局，

dataset：Structured3D Dataset 和NYU v2

## Neural Fields meet Explicit Geometric Representations for Inverse Rendering of Urban Scenes（CVPR 2023）

## LERF: Language Embedded Radiance Fields

## CLIP-NeRF: Text-and-Image Driven Manipulation of Neural Radiance Fields(2022CVPR)

## SINE: Semantic-driven Image-based NeRF Editing with Prior-guided Editing Field (2023.3 arxiv Zhejiang & Google)

可以对nerf进行语义编辑

## Intrinsic Indoor Scene Reconstruction and Editing via Raytracing in Neural SDFs （CVPR 2023）

## Neural Radiance Fields for Manhattan Scenes with Unknown   Manhattan Frame (2022 arxiv)

## Neural 3D Scene Reconstruction with the Manhattan-world Assumption (2022 CVPR oral)

基于曼哈顿世界假设的三维重建。



## CompoNeRF: Text-guided Multi-object Compositional NeRF with Editable 3D Scene Layout



## Nerflets: Local Radiance Fields for Efficient Structure-Aware   3D Scene Representation from 2D Supervision（arxiv 2023.3 Standford&Google）

设计NeRFflets，用多个local 的nerf来表示三维场景。能够更好地表示appearance, density, semantics, and object instance。从而可以更好地用在三维场景理解和全景图分割上面。

## LGT-Net: Indoor Panoramic Room Layout Estimation with Geometry-Aware Transformer Network(CVPR 2022 开源)

主要创新：1.将layout用horizon depth和屋子高度来表示 2. 将transformer引入到全景图的估计网络中。3. 设计了一个相对位置embedding的方法来增强对空间的感知。

## NeRF++

 SHAPE-RADIANCE AMBIGUITY

 INVERTED SPHERE PARAMETRIZATION

## Self-Supervised Monocular 3D Scene Reconstruction with Radiance Fields （arxiv 2022.12）

[SceneRF具有辐射场的自监督单目三维场景重建 - 腾讯云开发者社区-腾讯云](https://cloud.tencent.com/developer/article/2202523)

## Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields （Google 2023.4 ）



## 360Roam: Real-Time Indoor Roaming Using Geometry-Aware ${360^\circ}$ Radiance Fields （2022.8）

首先是定义360NeRF，实际上就是用全景图扩展到三维的坐标上。

创新点在于Progressive probabilities geometry estimation。具体来说就是先估计occupancy map：首先训练360NeRF，基于这个预测深度来构建一个八叉树的octomap，保存每个voxel的occupancy，实际上process是不断更新这个occupancy



# 多图Layout 估计



## 360-DFPE: Leveraging Monocular 360-Layouts for Direct Floor Plan Estimation （2022RAL）

## 360-MLC: Multi-view Layout Consistency for Self-training and Hyper-parameter Tuning （2022.10 国立清华大学）

利用了self-training的想法，用单个layout估计，然后进行register，得到对应target view的伪标签和uncertainty

亮点：如何确定伪标签？求这些layout的mean和std，从而设定一个weighted boundary consistency loss

![](C:\Users\bai\AppData\Roaming\marktext\images\2023-04-18-21-57-23-image.png)

下面的std实际上就是为了用uncertainty的思想对训练监督进行控制。

亮点2：由于没有多视角layout的GT，因此设计了一个layout consistency 来估计，实际上就是对floorplan进行熵loss计算。

## Mvlayoutnet: 3d layout reconstruction with multi-view panoramas （2021.3）

## Psmnet: Position-aware stereo merging network for room layout estimation （2022.3）



## GPR-Net: Multi-view Layout Estimation via a Geometry-aware Panorama Registration Network(2022.10)



# 多视图深度估计

## The Temporal Opportunist: Self-Supervised Multi-Frame Monocular Depth

[深度估计 ManyDepth 笔记_m_buddy的博客-CSDN博客](https://blog.csdn.net/m_buddy/article/details/117398459)

[知乎问答，关于cost volume](https://www.zhihu.com/question/366970399/answer/1115449421)
