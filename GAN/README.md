# GAN

## Table of Contents
- [Generative Adversarial Networks](#generative-adversarial-networks)
  - [2017.4](#2017-4)
- [2016.7](#2016.7)
- [2016.9](#2016.9)

## Generative Adversarial Networks

### 2017 4
**Machine Learning**
机器学习的模型可大体分为两类，生成模型（Generative Model）和判别模型（Discriminative Model）。
判别模型需要输入变量x ，通过某种模型来预测p(y|x) 。生成模型是给定某种隐含信息，来随机产生观测数据。

什么是机器学习？
一句话来概括就是，在训练过程中给予回馈，使得结果接近我们的期望。
* 对于分类问题（classification），我们希望loss在接近bound以后，就不要再有变化，所以我们选择交叉熵（Cross Entropy）作为回馈；
* 在回归问题（regression）中，我们则希望loss只有在两者一摸一样时才保持不变，所以选择点之间的欧式距离（MSE）作为回馈。

损失函数（回馈）的选择，会明显影响到训练结果的质量，是设计模型的重中之重。这五年来，神经网络的变种已有不下几百种，但损失函数却寥寥无几。
例如caffe的官方文档中，只提供了八种标准损失函数 Caffe | Layer Catalogue [CaffeLayerCatalogue](http://caffe.berkeleyvision.org/tutorial/layers.html)。

- 对于**判别模型**，损失函数是容易定义的，因为输出的目标相对简单。
- 但对于**生成模型**，损失函数的定义就不是那么容易。
例如:
  - 对于NLP方面的生成语句，虽然有BLEU这一优秀的衡量指标，但由于难以求导，以至于无法放进模型训练；
  - 对于生成猫咪图片的任务，如果简单地将损失函数定义为“和已有图片的欧式距离”，那么结果将是数据库里图片的诡异混合，效果惨不忍睹。

当我们希望神经网络画一只猫的时候，显然是希望这张图有一个动物的轮廓、带质感的毛发、和一个霸气的眼神，而不是冷冰冰的欧式距离最优解。如何将我们对于猫的期望放到模型中训练呢？这就是GAN的Adversarial部分解决的问题。

**Adversarial：对抗（互怼 ）**

在generative部分提到了，我们对于猫（生成结果）的期望，往往是一个暧昧不清，难以数学公理化定义的范式。但等一下，说到处理暧昧不清、难以公理化的问题，之前提到的判别任务不也是吗？比如图像分类，一堆RGB像素点和最后N类别的概率分布模型，显然是无法从传统数学角度定义的。那为何，不把**生成模型的回馈部分**，交给判别模型呢？这就是Goodfellow天才般的创意--他将机器学习中的两大类模型，**Generative和Discrimitive**给紧密地联合在了一起。

模型一览

![gan](./image/1.png)

对抗生成网络主要由生成部分G，和判别部分D组成。
训练过程描述如下：

1. 输入噪声（隐藏变量）z
2. 通过生成部分G 得到x_{fake}=G(z)
3. 从真实数据集中取一部分真实数据x_{real}
4. 将两者混合x=x_{fake} + x_{real}
5. 将数据喂入判别部分D ，给定标签x_{fake}=0,x_{real}=1 （简单的二类分类器）
6. 按照分类结果，回传loss

在整个过程中，D要尽可能的使D(G(z))=0，D(x_{real})=1（火眼晶晶，不错杀也不漏杀）。而G则要使得D(G(z))=1，**即让生成的图片尽可能以假乱真**。整个训练过程就像是两个玩家在相互对抗，也正是这个名字Adversarial的来源。在论文中[1406.2661] [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)，Goodfellow从理论上证明了该算法的收敛性，以及在模型收敛时，生成数据具有和真实数据相同的分布（保证了模型效果）。

从研究角度，GAN给众多生成模型提供了一种新的训练思路，催生了许多后续作品。例如根据自己喜好定制二次元妹子（逃），根据文字生成对应描述图片（[Newmu/dcgan_code](https://github.com/Newmu/dcgan_code), [hanzhanggit/StackGAN](https://github.com/hanzhanggit/StackGAN))，甚至利用标签生成3D宜家家居模型（[zck119/3dgan-release](https://github.com/zck119/3dgan-release)），这些作品的效果无一不令人惊叹。同时，难人可贵的是这篇论文有很强的数学论证，不同于前几年的套模型的结果说话，而是从理论上保证了模型的可靠性。虽然目前训练还时常碰到困难，后续已有更新工作改善该问题（WGAN, Loss Sensetive GAN, Least Square GAN)，相信终有一日能克服。

### 2017 4