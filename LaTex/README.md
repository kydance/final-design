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

@article{haut2021distributed,
  title={Distributed deep learning for remote sensing data interpretation},
  author={Haut, Juan M and Paoletti, Mercedes E and Moreno-{\'A}lvarez, Sergio and Plaza, Javier and Rico-Gallego, Juan-Antonio and Plaza, Antonio},
  journal={Proceedings of the IEEE},
  volume={109},
  number={8},
  pages={1320--1349},
  year={2021},
  publisher={IEEE}
}

@article{quirita2016new,
  title={A new cloud computing architecture for the classification of remote sensing data},
  author={Quirita, Victor Andres Ayma and da Costa, Gilson Alexandre Ostwald Pedro and Happ, Patrick Nigri and Feitosa, Raul Queiroz and da Silva Ferreira, Rodrigo and Oliveira, D{\'a}rio Augusto Borges and Plaza, Antonio},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume={10},
  number={2},
  pages={409--416},
  year={2016},
  publisher={IEEE}
}

@article{sun2019efficient,
  title={An efficient and scalable framework for processing remotely sensed big data in cloud computing environments},
  author={Sun, Jin and Zhang, Yi and Wu, Zebin and Zhu, Yaoqin and Yin, Xianliang and Ding, Zhongzheng and Wei, Zhihui and Plaza, Javier and Plaza, Antonio},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={57},
  number={7},
  pages={4294--4308},
  year={2019},
  publisher={IEEE}
}

@article{lunga2020apache,
  title={Apache spark accelerated deep learning inference for large scale satellite image analytics},
  author={Lunga, Dalton and Gerrand, Jonathan and Yang, Lexie and Layton, Christopher and Stewart, Robert},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume={13},
  pages={271--283},
  year={2020},
  publisher={IEEE}
}

<!-- TAG 已添加 -->
@article{chi2016big,
  title={Big data for remote sensing: Challenges and opportunities},
  author={Chi, Mingmin and Plaza, Antonio and Benediktsson, Jon Atli and Sun, Zhongyi and Shen, Jinsheng and Zhu, Yangyong},
  journal={Proceedings of the IEEE},
  volume={104},
  number={11},
  pages={2207--2219},
  year={2016},
  publisher={IEEE}
}

@article{haut2019cloud,
  title={Cloud deep networks for hyperspectral image analysis},
  author={Haut, Juan Mario and Gallardo, Jose Antonio and Paoletti, Mercedes E and Cavallaro, Gabriele and Plaza, Javier and Plaza, Antonio and Riedel, Morris},
  journal={IEEE transactions on geoscience and remote sensing},
  volume={57},
  number={12},
  pages={9832--9848},
  year={2019},
  publisher={IEEE}
}

<!-- TAG 已添加 -->
@inproceedings{song2021communication,
  title={Communication efficient SGD via gradient sampling with Bayes prior},
  author={Song, Liuyihan and Zhao, Kang and Pan, Pan and Liu, Yu and Zhang, Yingya and Xu, Yinghui and Jin, Rong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12065--12074},
  year={2021}
}

<!-- TAG 已添加 -->
@article{ghosh2021communication,
  title={Communication-efficient and byzantine-robust distributed learning with error feedback},
  author={Ghosh, Avishek and Maity, Raj Kumar and Kadhe, Swanand and Mazumdar, Arya and Ramchandran, Kannan},
  journal={IEEE Journal on Selected Areas in Information Theory},
  volume={2},
  number={3},
  pages={942--953},
  year={2021},
  publisher={IEEE}
}

<!-- TAG 已添加 -->
@inproceedings{zhang2020communication,
  title={Communication-efficient network-distributed optimization with differential-coded compressors},
  author={Zhang, Xin and Liu, Jia and Zhu, Zhengyuan and Bentley, Elizabeth S},
  booktitle={IEEE INFOCOM 2020-IEEE Conference on Computer Communications},
  pages={317--326},
  year={2020},
  organization={IEEE}
}

<!-- TAG 已添加 -->
@article{khirirat2020compressed,
  title={Compressed gradient methods with hessian-aided error compensation},
  author={Khirirat, Sarit and Magn{\'u}sson, Sindri and Johansson, Mikael},
  journal={IEEE Transactions on Signal Processing},
  volume={69},
  pages={998--1011},
  year={2020},
  publisher={IEEE}
}

@article{adikari2021compressing,
  title={Compressing gradients by exploiting temporal correlation in momentum-SGD},
  author={Adikari, Tharindu B and Draper, Stark C},
  journal={IEEE Journal on Selected Areas in Information Theory},
  volume={2},
  number={3},
  pages={970--986},
  year={2021},
  publisher={IEEE}
}

@article{abrahamyan2021learned,
  title={Learned gradient compression for distributed deep learning},
  author={Abrahamyan, Lusine and Chen, Yiming and Bekoulis, Giannis and Deligiannis, Nikos},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  volume={33},
  number={12},
  pages={7330--7344},
  year={2021},
  publisher={IEEE}
}

@article{du2021parallel,
  title={Parallel and distributed computing for anomaly detection from hyperspectral remote sensing imagery},
  author={Du, Qian and Tang, Bo and Xie, Weiying and Li, Wei},
  journal={Proceedings of the IEEE},
  volume={109},
  number={8},
  pages={1306--1319},
  year={2021},
  publisher={IEEE}
}

@article{liang2021pruning,
  title={Pruning and quantization for deep neural network acceleration: A survey},
  author={Liang, Tailin and Glossner, John and Wang, Lei and Shi, Shaobo and Zhang, Xiaotong},
  journal={Neurocomputing},
  volume={461},
  pages={370--403},
  year={2021},
  publisher={Elsevier}
}

@article{wu2021recent,
  title={Recent developments in parallel and distributed computing for remotely sensed big data processing},
  author={Wu, Zebin and Sun, Jin and Zhang, Yi and Wei, Zhihui and Chanussot, Jocelyn},
  journal={Proceedings of the IEEE},
  volume={109},
  number={8},
  pages={1282--1305},
  year={2021},
  publisher={IEEE}
}
