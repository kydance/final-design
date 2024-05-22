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

- [ ] 论文中提出了两种梯度压缩算法，这两中算法相互之间的关系是什么？分别有什么优缺点？
- [x] P15 页中2080Ti采用Turing架构，不是Truing架构。
- [x] P40页表3.4中GCAC在100x大压缩比时，性能没有表现出明显的性能优势，论文只是归因于“近似质心”的加速计算技术，请分析其具体原因。
- [ ] 针对高光谱目标检测和自然图像分类，k-RNGC梯度压缩算法对其影响是否一致？有何不同影响？
- [ ] 图2.4、图3.6、图3.7、图4.5显示不全
- [ ] 图 3.5 中,较难区分孰优孰劣,建议在图中通过指示性图标标出方法的优越之处
- [ ] 在算法 4.2 中,binaryTraversSearch 函数是什么?如是已有方法,请给出参考文献
