---
library_name: setfit
metrics:
- accuracy
pipeline_tag: text-classification
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: 张学友 动画片 中新网 张学友 新人王 新人王 张学友 忍不住
- text: 高跟鞋 旱冰鞋 中新网 克林特 特伍德 安吉丽娜 化身为 高跟鞋 旱冰鞋 安吉丽娜 高跟鞋 旱冰鞋 老照片 高跟鞋 旱冰鞋
- text: 伍德斯 大杂烩 伍德斯 奥斯卡 大杂烩 改编自 奥斯卡 福克斯 鱼鱼文
- text: 体育讯 季后赛 半决赛 波士顿 凯尔特人 加时赛 迈阿密 大比分 波士顿 德维恩 不合理 更衣室 凯尔特人 结构性 战斗力 凯尔特人 没想到 成绩单
    精神力量 压制住 不逊于 第一节 加内特 令人吃惊 第一节 勒布朗 詹姆斯 詹姆斯 詹姆斯 没想到 詹姆斯 詹姆斯 不用说 下半场 詹姆斯 第四节 事实上
    第三节 第四节 即便如此 凯尔特人
- text: 体育讯 季后赛 雄鹿队 总经理 哈蒙德 哈蒙德 年轻人 总经理 达雷尔莫雷 第一次 布鲁克斯 太阳队
inference: true
---

# SetFit

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
<!-- - **Sentence Transformer:** [Unknown](https://huggingface.co/unknown) -->
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 128 tokens
- **Number of Classes:** 2 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label | Examples                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|:------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 娱乐    | <ul><li>'斯图尔特 克里斯 斯图尔特 家门口 大麻烟 年轻一代 斯图尔特 洛杉矶 家门口 斯图尔特 家门口 大麻烟 斯图尔特 斯图尔特 福斯特 丹尼斯 凯瑟琳 凯瑟琳 结果显示 凯瑟琳 总收入 人民币 人民币 好莱坞 麦考利 麦考利 模特儿 可卡因 可卡因 鱼鱼文'</li><li>'七龙珠 媒体报道 真人版 七龙珠 万美元 意味着 七龙珠 可能性 孙悟空 詹姆斯 斯特斯 七龙珠 真人版 第一手 斯特斯 七龙珠 是因为 第二部 有意思 有意思 第二部 七龙珠 七龙珠 第一部 七龙珠 第一部 第一部 第一部 第一部 一部分 第二部 七龙珠 事实上 第二部 七龙珠 七龙珠 心之洛文'</li><li>'安吉丽娜 好莱坞 安吉丽娜 布莱德 返老还童 记者会 金像奖 安吉丽娜 福克斯 安吉丽娜 好莱坞 埃夫隆 哈金斯 嘉年华 运动员 俐俐文'</li></ul>                                                                                                                                                                                                                                                                                       |
| 体育    | <ul><li>'体育讯 常规赛 常规赛 总冠军 圣安东尼奥 常规赛 总冠军 奥拉朱旺 奥尼尔 第二个 可能性 微乎其微 即便如此 季后赛 得分手 常规赛 詹姆斯 分波什 凯尔特人 皮尔斯 约翰逊 克劳福德 诺维茨基 杜兰特 分灰熊 兰多夫 加索尔 万劫不复 东西部 英雄主义'</li><li>'马布里 体育讯 常规赛 第二十六 辽宁队 两连胜 辽宁队 麦尔斯 马布里 刘书楠 辽宁队 有的放矢 杨文博 辽宁队 李晓旭 马布里 紧跟着 马布里 辽宁队 不甘示弱 李晓旭 锦上添花 两分钟 三分球 第二节 三分钟 辽宁队 杨文博 辽宁队 麦尔斯 马布里 上半场 辽宁队 下半场 辽宁队 李晓旭 先拔头筹 紧跟着 麦尔斯 进一步 麦尔斯 马布里 辽宁队 马布里 三分球 辽宁队 收效甚微 关键时刻 李晓旭 上篮得分 稳住阵脚 两分钟 马布里 个位数 辽宁队 第四节 杨文博 有的放矢 辽宁队 紧跟着 马布里 辽宁队 上篮得分 关键时刻 李晓旭 辽宁队 辽宁队 刘书楠 李晓旭 麦尔斯 马布里 杨文博'</li><li>'体育讯 迈阿密 总比分 第二轮 凯尔特人 系列赛 展现出 统治力 第一节 两位数 如果说 埃里克 斯特拉 事实上 第一节 米克斯 两位数 安东尼 马里奥 安东尼 不逊于 三分球 命中率 是因为 大多数 勒布朗 詹姆斯 德维恩 安东尼 第四节 安东尼 安东尼 心理素质 安东尼 也就是说 凯尔特人 安东尼 凯尔特人 安东尼 凯尔特人 系列赛 凯尔特人 第一次 卡洛斯 阿罗约 詹姆斯 安东尼'</li></ul> |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("张学友 动画片 中新网 张学友 新人王 新人王 张学友 忍不住")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median  | Max |
|:-------------|:----|:--------|:----|
| Word count   | 1   | 32.1475 | 192 |

| Label | Training Sample Count |
|:------|:----------------------|
| 体育    | 402                   |
| 娱乐    | 398                   |

### Training Hyperparameters
- batch_size: (6, 6)
- num_epochs: (10, 10)
- max_steps: -1
- sampling_strategy: oversampling
- num_iterations: 10
- body_learning_rate: (2e-05, 1e-05)
- head_learning_rate: 0.01
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: True

### Training Results
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0004 | 1    | 0.1886        | -               |
| 0.0187 | 50   | 0.2006        | -               |
| 0.0375 | 100  | 0.1118        | -               |
| 0.0562 | 150  | 0.0792        | -               |
| 0.0750 | 200  | 0.0192        | -               |
| 0.0937 | 250  | 0.0042        | -               |
| 0.1125 | 300  | 0.0048        | -               |
| 0.1312 | 350  | 0.0018        | -               |
| 0.1500 | 400  | 0.0009        | -               |
| 0.1687 | 450  | 0.0007        | -               |
| 0.1875 | 500  | 0.0008        | -               |
| 0.2062 | 550  | 0.0003        | -               |
| 0.2250 | 600  | 0.0003        | -               |
| 0.2437 | 650  | 0.0001        | -               |
| 0.2625 | 700  | 0.0001        | -               |
| 0.2812 | 750  | 0.0001        | -               |
| 0.3000 | 800  | 0.0001        | -               |
| 0.3187 | 850  | 0.0001        | -               |
| 0.3375 | 900  | 0.0001        | -               |
| 0.3562 | 950  | 0.0           | -               |
| 0.3750 | 1000 | 0.0001        | -               |
| 0.3937 | 1050 | 0.0001        | -               |
| 0.4124 | 1100 | 0.0001        | -               |
| 0.4312 | 1150 | 0.0           | -               |
| 0.4499 | 1200 | 0.0           | -               |
| 0.4687 | 1250 | 0.0           | -               |
| 0.4874 | 1300 | 0.0           | -               |
| 0.5062 | 1350 | 0.0           | -               |
| 0.5249 | 1400 | 0.0           | -               |
| 0.5437 | 1450 | 0.0           | -               |
| 0.5624 | 1500 | 0.0           | -               |
| 0.5812 | 1550 | 0.0           | -               |
| 0.5999 | 1600 | 0.0           | -               |
| 0.6187 | 1650 | 0.0           | -               |
| 0.6374 | 1700 | 0.0           | -               |
| 0.6562 | 1750 | 0.0           | -               |
| 0.6749 | 1800 | 0.0           | -               |
| 0.6937 | 1850 | 0.0           | -               |
| 0.7124 | 1900 | 0.0           | -               |
| 0.7312 | 1950 | 0.0           | -               |
| 0.7499 | 2000 | 0.0           | -               |
| 0.7687 | 2050 | 0.0           | -               |
| 0.7874 | 2100 | 0.0           | -               |
| 0.8061 | 2150 | 0.0           | -               |
| 0.8249 | 2200 | 0.0           | -               |
| 0.8436 | 2250 | 0.0           | -               |
| 0.8624 | 2300 | 0.0           | -               |
| 0.8811 | 2350 | 0.0           | -               |
| 0.8999 | 2400 | 0.0           | -               |
| 0.9186 | 2450 | 0.0           | -               |
| 0.9374 | 2500 | 0.0           | -               |
| 0.9561 | 2550 | 0.0           | -               |
| 0.9749 | 2600 | 0.0           | -               |
| 0.9936 | 2650 | 0.0           | -               |
| 1.0    | 2667 | -             | 0.0086          |

### Framework Versions
- Python: 3.10.15
- SetFit: 1.0.3
- Sentence Transformers: 3.1.1
- Transformers: 4.40.2
- PyTorch: 2.4.0+cu121
- Datasets: 3.0.2
- Tokenizers: 0.19.1

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->