---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: topicm
  language: python
  name: python3
---

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AnFreTh/STREAM/blob/bugfixes/docs/notebooks/quickstart.ipynb)
[![Open On GitHub](https://img.shields.io/badge/Open-on%20GitHub-blue?logo=GitHub)](https://github.com/AnFreTh/STREAM/blob/bugfixes/docs/notebooks/quickstart.ipynb)

# Quickstart

```{code-cell} ipython3
from stream_topic.models import CEDC, DCTE
from stream_topic.utils import TMDataset


import warnings
warnings.filterwarnings("ignore")
```

## CEDC model

```{code-cell} ipython3
dataset = TMDataset()
dataset.fetch_dataset("DummyDataset")
```

```{code-cell} ipython3
model = CEDC(num_topics=10)
output = model.fit(dataset)
```

```{code-cell} ipython3
from stream_topic.visuals import visualize_topic_model, visualize_topics

visualize_topic_model(
    model, 
    reduce_first=True, 
    port=8052,
    )
```

## KMeansTM model

```{code-cell} ipython3
from stream_topic.models import KmeansTM
model = KmeansTM(num_topics=10)
output = model.fit(dataset)
```

```{code-cell} ipython3
visualize_topic_model(
    model, 
    reduce_first=True, 
    port=8053,
    )
```

```{code-cell} ipython3

```
