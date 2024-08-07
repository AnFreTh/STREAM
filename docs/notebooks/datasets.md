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

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AnFreTh/STREAM/blob/develop/docs/notebooks/datasets.ipynb)
[![Open On GitHub](https://img.shields.io/badge/Open-on%20GitHub-blue?logo=GitHub)](https://github.com/AnFreTh/STREAM/blob/develop/docs/notebooks/datasets.ipynb)

# Datasets

+++

The dataset module provides and easy way to load and preprocess the datasets. The package comes with a few datasets that are commonly used in topic modleing research. The datasets are:

    - 20NewsGroup
    - BBC_News
    - Stocktwits_GME
    - Reddit_GME'
    - Reuters'
    - Spotify
    - Spotify_most_popular
    - Poliblogs
    - Spotify_least_popular

Please see the functionalities availabe in the `TMDataset` module.

```{code-cell} ipython3
from stream_topic.utils import TMDataset

import warnings
warnings.filterwarnings("ignore")
```

## Using default datasets

- these datasets are already preprocessed and ready to be used for topic modeling
- these datasets are included in the package and can be loaded using the `TMDataset` module

```{code-cell} ipython3
dataset = TMDataset()
dataset.get_dataset_list()
```

```{code-cell} ipython3
dataset.fetch_dataset(name="Reuters")
```

```{code-cell} ipython3
dataset.get_bow()
```

```{code-cell} ipython3
dataset.get_tfidf()
```

```{code-cell} ipython3
# dataset.get_word_embeddings()
```

```{code-cell} ipython3
dataset.fetch_dataset('Spotify')
```

```{code-cell} ipython3
dataset.dataframe.head()
```

```{code-cell} ipython3
dataset.texts[:2]
```

```{code-cell} ipython3
dataset.tokens
```

```{code-cell} ipython3
dataset.labels[:5]
```

## Loading own dataset

```{code-cell} ipython3
from stream_topic.utils import TMDataset

import warnings
warnings.filterwarnings("ignore")
```

```{code-cell} ipython3
import pandas as pd
import numpy as np


# Simulating some example data
np.random.seed(0)

# Generate 1000 random strings of lengths between 1 and 5, containing letters 'A' to 'Z'
random_documents = [''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 
                                             np.random.randint(1, 6))) for _ in range(1000)]

# Generate 1000 random labels from 1 to 4 as strings
random_labels = np.random.choice(['1', '2', '3', '4'], 1000)

# Create DataFrame
my_data = pd.DataFrame({"Documents": random_documents, "Labels": random_labels})
```

```{code-cell} ipython3
dataset = TMDataset()
dataset.create_load_save_dataset(
    data=my_data, 
    dataset_name="sample_data",
    save_dir="data/",
    doc_column="Documents",
    label_column="Labels"
    )
```

```{code-cell} ipython3
# the new data is saved in the data folder unlike the default datasets which are saved in package directory under preprocessed_data folder.
# therefore, you need to provide the path to the data folder to fetch the dataset
dataset.fetch_dataset(name="sample_data", dataset_path="data/")
```

```{code-cell} ipython3
dataset.dataframe.head()
```

```{code-cell} ipython3

```
