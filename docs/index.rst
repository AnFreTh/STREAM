.. STREAM documentation master file, created by
   sphinx-quickstart on Thu Jul 18 13:02:06 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
******************************
STREAM
******************************
.. div:: sd-text-left sd-font-italic

   We present STREAM, a Simplified Topic Retrieval, Exploration, and Analysis Module for user-friendly topic modelling and especially subsequent interactive topic visualization and analysis. For better topic analysis, we implement multiple intruder-word based topic evaluation metrics. Additionally, we publicize multiple new datasets that can extend the so far very limited number of publicly available benchmark datasets in topic modeling. We integrate downstream interpretable analysis modules to enable users to easily analyse the created topics in downstream tasks together with additional tabular information.

Features
^^^^^^^^^

.. grid::

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Topics Models
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Stream provides a wide range of topic models like LDA, NMF, LSA, BERTopic, etc. to help you extract topics from your text data.

   .. grid-item::
      :columns: 12 12 12 6

      .. card:: Functional API
         :class-card: sd-border-0
         :shadow: none
         :class-title: sd-fs-5

         .. div:: sd-font-normal

            Stream's functional api makes it user-friendly to use and easy to integrate with other libraries. Blabla bla bla

----


Installation
^^^^^^^^^^^^

It is recommended to install the latest version of STREAM. You can install the latest version of STREAM using the following command:

.. code-block:: bash

   pip install -U stream


You could also install from source by cloning the repository and running the following command:

.. code-block:: bash

   git clone https://github.com/AnFreTh/STREAM
   cd stream
   pip install -e .

----


Basic usage
^^^^^^^^^^^^

Simple example of using STREAM to extract topics from a text dataset and visualize the topics:

.. testcode::

   # import the necessary modules
   from stream.utils import TMDataset
   from stream.models import CEDC
   from stream.visuals import visualize_topic_model


   # fetch the dataset
   dataset = TMDataset()
   dataset.fetch_dataset("Spotify_random")

   # create and train the model
   model = CEDC(num_topics=5)  # Create model
   model_output = model.train_model(dataset) 

   # visualize the model results
   visualize_topic_model(model, port=8053)

----


Learn more
^^^^^^^^^^

.. grid::

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`rocket_launch;2em` Quickstart
         :class-card: sd-text-black sd-bg-light
         :link: quick_start.html

   .. grid-item::
      :columns: 6 6 6 4

      .. card:: :material-regular:`library_books;2em` Guides
         :class-card: sd-text-black sd-bg-light
         :link: guides/index.html
         
----

.. toctree::
   :name: Getting Started
   :caption: Getting Started
   :maxdepth: 2
   :hidden:

   installation
   

.. toctree::
   :name: API Reference
   :caption: API Reference
   :maxdepth: 2
   :hidden:
   
   api/stream.models/index
   api/stream.NAM/index
   api/stream.metrics/index
   api/stream.preprocessor/index
   api/stream.visuals/index
   api/stream.utils/index


.. toctree::
   :name: Developer Guide
   :caption: Developer Guide
   :maxdepth: 1
   :hidden: