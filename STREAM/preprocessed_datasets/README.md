## Available Datasets
-------------------

You can leverage and load all dataset available in [OCTIS](https://aclanthology.org/2021.eacl-demos.31.pdf).

Additionally there are some preprocessed datasets available in STREAM:

Available Datasets
=================

| Name                  | # Docs  | # Words | # Features | Description                                                                                      |
| --------------------- | ------- | ------- | ---------- | ------------------------------------------------------------------------------------------------ |
| Spotify_most_popular  | 4,538   | 53,181  | 14         | Spotify dataset comprised of popular song lyrics and various tabular features.                   |
| Spotify_least_popular | 4,374   | 111,738 | 14         | Spotify dataset comprised of less popular song lyrics and various tabular features.              |
| Spotify               | 4,185   | 80,619  | 14         | General Spotify dataset with song lyrics and various tabular features.                           |
| Reddit_GME            | 21,549  | 21,309  | 6          | Reddit dataset filtered for "Gamestop" (GME) from the Subreddit "r/wallstreetbets".              |
| Stocktwits_GME        | 11,114  | 19,383  | 3          | Stocktwits dataset filtered for "Gamestop" (GME), covering the GME short squeeze of 2021.        |
| Stocktwits_GME_large  | 136,138 | 80,435  | 3          | Larger Stocktwits dataset filtered for "Gamestop" (GME), covering the GME short squeeze of 2021. |
| Reuters               | 8,929   | 24,803  | -          | Preprocessed Reuters dataset well suited for comparing topic model outputs.                      |
| Poliblogs             | 13,246  | 70,726  | 4          | Preprocessed Poliblogs dataset well suited for comparing topic model outputs.                    |
|                       |



Load a preprocessed dataset
----------------------------

To load one of the already preprocessed datasets as follows:


```python
    from STREAM.data_utils import TMDataset
   
   dataset = TMDataset()
   dataset.fetch_dataset("Spotify")
```

Just use one of the dataset names listed above or one of the datasets from octis. Note that they are case-sensitive.


Load a custom preprocessed dataset
----------------------------

Otherwise, you can load a custom preprocessed dataset in the following way, by simply using a pandas dataframe:

```python
   from STREAM.data_utils import TMDataset

   dataset = TMDataset()
   dataset = dataset.create_load_save_dataset(my_data, "test",
        "..",
        doc_column="Documents",
        label_column="Labels",
        )
```

