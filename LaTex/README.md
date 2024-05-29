# Problems And Answers

```bash
sudo pacman -S texlive-core texlive-formatsextra texlive-mathscience texlive-langchinese texlive-latexextra texlive-bibtexextra biber
```

- ! LaTeX Error: File `ulem.sty` is not found:

    `texlive-formatsextra`

- ! LaTeX Error: File `algpseudocode.sty' not found.

    `texlive-mathscience`

- ! LaTeX Error: File `fontspec.sty' not found.

    `texlive-latexextra`

- ! LaTeX Error: File `biblatex.sty' not found.

    `texlive-bibtexextra`

- ! Token not allowed in a PDF string (Unicode): (hyperref) removing `math shift’.

  `\subsection{$k$-互异近邻梯度选择算法}`
    ->
  `\subsection{\texorpdfstring{$k$}.-互异近邻梯度选择算法}`

`inkscape --export-type="pdf" 图片名.svg`

## Todo-List

- [x]  第三章，图3.1、图3.2 修改 - images/outlines/g2.pptx

- [x] 修复关键警告

- [x] acc 和 loss 改成小图

- [x] Update loss figures.

- [x] 第四章图4.1，修改图例

- [x] 公式、图、表 `ref{}` 前后添加一个空格

- [x] 校对各章中的算法框架

- [x] 注意后面图中的Top-k分类精度：$acc_{k}$

- [x] 根据中文摘要，修改英文摘要

- [x] 符号对照表

- [x] 缩略语对照表

- [x] 加入网络拓扑节点数量相关描述

- [x] 图2.1中的$\delta$应修改为$\sigma$，用以表示激活函数

- [x] 第三章中Docker镜像描述时，版本号应控制一致

- [x] 表3.4标题，“与其他梯度算法”修改为“与其他算法”

- [x] 图2.6中，将公式中的“Loss”修改为$\mathcal{L}$

- [x] 图2.6中，将公式中的“K”修改为N，表示网络拓扑中Worker的数量

- [x] 科研项目脱敏

- [x] 加入模型描述$J$到对照表中

- [x] 将公式中的 “N” 尝试更换字母，表示网络拓扑中Worker的数量

- [x] 参考文献， 修复“[j]”标志

- [ ] 修改整体算法名称：一般不叫基于  比如什么引导 什么驱动

## 评审专家质询的问题

> 详细答案见答辩 PPT

### 吴宪云

- [x] 论文中提出了两种梯度压缩算法，这两中算法相互之间的关系是什么？分别有什么优缺点？
- [x] P15 页中 2080Ti 采用 Turing 架构，不是 Truing 架构。
- [x] P40页表3.4中GCAC在100x大压缩比时，性能没有表现出明显的性能优势，论文只是归因于“近似质心”的加速计算技术，请分析其具体原因。
- [x] 针对高光谱目标检测和自然图像分类，k-RNGC梯度压缩算法对其影响是否一致？有何不同影响？

### 宋焕生

- [x] 论文写作和语言表达上需要进一步简练。
- [x] 一些排版问题：图 2.4 中，显示补全，看不到（a-b）；图 3.6、图 3.7、图 4.5 中，也存在上述问题。
- [x] 图 3.5 中，较难区分孰优孰劣，建议在图中通过指示性图标标出方法的优越之处。
- [x] 在算法 4.2 中，binaryTraversSearch 函数是什么?如是已有方法，请给出参考文献。

  已添加相关引用：`CORMEN T H, LEISERSON C E, RIVEST R L, et al. Introduction to algorithms[M]. MIT press, 2022.`

- [x] 在网络训练过程中，如 3.7 中，论文是否固定了 seed 的值进行训练？

  默认 seed 值固定，但可通过提供的参数进行修改 seed 值。

- [x] 训练网络中，所涉及到的 Batch、Epoch 等高参是如何确定？

  1、Batch Size 参数确定考虑因素如下：

  - 在计算机科学和工程领域，参数往往采用 2 的幂次（例如 16, 32, 64, 128等），能够确保内存地址和数据在内存中的对齐，可以提高访问效率，同时也有助于提高网络协议和文件系统中的数据传输和存储效率。
  - 为了能使分布式节点上的 GPU 利用率最大化，尽量选取大 Batch Size。
  - 在满足以上两点的情况下，还需要考虑 Batch Size 对网络模型泛化性的影响（Batch Size 过大可能导致泛化性下降）。

  2、Epoch 参数确定过程如下：

  首先，选取一个较大的值进行模型训练；然后，根据训练过程中得到的评价指标-Epochs曲线和 TrainingLoss-Epochs 曲线确定大致数值，确保网络模型得到充分训练和学习，同时又不浪费计算资源。

- [x] k-RNGC 中的高参是如何选取的？

- [x] 在表 4.5 中，只列出一种方法（本文所提出的），为什么不加入其他方法进行对比？此外却缺少耗时分析。

  1、由于 GCAC 算法和 k-RNGC 算法同样都在 CIFAR10 和 AeroRIT、Xiongan New Aera（Matiwan Village）数据集上进行实验验证，而在第三章已将其他对比方法的结果列出，为了避免重复，故第四章中仅列出k-RNGC 算法的实验结果。

  2、第四章已将“耗时分析”实验小节的位置调整，与第三章保持一致。
