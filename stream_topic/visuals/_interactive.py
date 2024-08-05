import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import umap.umap_ as umap
from dash import Input, Output, dcc, html
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

from ..models.abstract_helper_models.mixins import SentenceEncodingMixin


def calculate_distances_to_other_topics(selected_topic_index, plot_df, model):
    """
    Calculate cosine distances between the selected topic and all other topics,
    along with their top 3 words.

    Parameters:
    -----------
    selected_topic_index : int
        Index of the selected topic.
    plot_df : pandas.DataFrame
        DataFrame containing the positions of topics on the plot, with 'x' and 'y' columns.
    model : model class of BaseModel
        Output dictionary from the topic modeling model, containing topic information.

    Returns:
    --------
    list of tuples
        List of tuples containing the index of the topic, its cosine distance to the
        selected topic, and the top 3 words associated with that topic.
    """
    selected_topic_position = plot_df.iloc[selected_topic_index][["x", "y"]]
    distances = []

    for index, row in plot_df.iterrows():
        if index != selected_topic_index:
            other_topic_position = row[["x", "y"]]
            distance = cosine(
                selected_topic_position, other_topic_position
            )  # Cosine distance
            top_words = ", ".join(
                [word for word, _ in model.topic_dict[index][:3]]
            )  # Top 3 words
            distances.append((index, distance, top_words))

    # Sort by distance
    distances.sort(key=lambda x: x[1])
    return distances


def _visualize_topic_model_2d(
    model,
    reduce_first=False,
    reducer="umap",
    port=8050,
    dataset=None,
    encoder_model="paraphrase-MiniLM-L3-v2",
    use_average=True,
    embeddings_path=None,
    embeddings_file_path=None,
):
    """
    Visualize a topic model in 2D space using UMAP, t-SNE, or PCA dimensionality reduction techniques.

    Parameters:
    -----------
    model : object
        Topic modeling model object.
    reduce_first : bool, optional
        Whether to reduce dimensions of embeddings first (default is False).
    reducer : str, optional
        Dimensionality reduction technique to use, one of ['umap', 'tsne', 'pca'] (default is 'umap').
    port : int, optional
        Port number for running the visualization dashboard (default is 8050).

    Returns:
    --------
    None
        The function launches a Dash server to visualize the topic model.
    """
    if not hasattr(model, "embeddings"):
        if dataset.has_embeddings(encoder_model):
            embeddings = dataset.get_embeddings(
                encoder_model,
                embeddings_path,
                embeddings_file_path,
            )
        else:
            encoder = SentenceEncodingMixin()
            embeddings = encoder.encode_documents(
                dataset.texts, encoder_model=encoder_model, use_average=use_average
            )
    else:
        embeddings = model.embeddings

    num_docs_per_topic = pd.Series(model.labels).value_counts().sort_index()

    # Extract top words for each topic with importance and format them vertically
    top_words_per_topic = {
        topic: "<br>".join(
            [f"{word} ({importance:.2f})" for word, importance in words[:5]]
        )
        for topic, words in model.topic_dict.items()
    }

    if reducer == "umap":
        reducer = umap.UMAP(n_components=2)
    elif reducer == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, learning_rate=200)
    elif reducer == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("reducer must be in ['umap', 'tnse', 'pca']")

    if reduce_first:
        embeddings = reducer.fit_transform(embeddings)
        topic_data = []
        # Iterate over unique labels and compute mean embedding for each
        for label in np.unique(model.labels):
            # Find embeddings corresponding to the current label
            label_embeddings = embeddings[model.labels == label]
            # Compute mean embedding for the current label
            mean_embedding = np.mean(label_embeddings, axis=0)
            # Store the mean embedding in the dictionary
            topic_data.append(mean_embedding)
    else:
        if hasattr(model, "topic_centroids"):
            topic_data = model.topic_centroids
        else:
            topic_data = []
            # Iterate over unique labels and compute mean embedding for each
            for label in np.unique(model.labels):
                # Find embeddings corresponding to the current label
                label_embeddings = embeddings[model.labels == label]
                # Compute mean embedding for the current label
                mean_embedding = np.mean(label_embeddings, axis=0)
                # Store the mean embedding in the dictionary
                topic_data.append(mean_embedding)

    topic_embeddings_2d = reducer.fit_transform(topic_data)

    # Create DataFrame for plotting
    plot_df = pd.DataFrame(topic_embeddings_2d, columns=["x", "y"])
    plot_df["topic"] = list(top_words_per_topic.keys())
    plot_df["num_docs"] = num_docs_per_topic
    plot_df["top_words"] = list(top_words_per_topic.values())

    app = dash.Dash(__name__)

    app.layout = html.Div(
        [
            html.H1("Lavish Topic Visualization"),
            dcc.Dropdown(
                id="side-dropdown",
                options=[
                    {"label": "Top Words", "value": "top_words"},
                    {"label": "Topic Distances", "value": "topic_distances"},
                ],
                value="top_words",
            ),
            dcc.Graph(id="main-plot"),
            dcc.Graph(id="side-plot"),
            html.Div(id="details-panel", children="Click a point for details"),
        ]
    )

    @app.callback(Output("main-plot", "figure"), [Input("side-dropdown", "value")])
    def update_main_plot(selected_option):
        fig = px.scatter(
            plot_df,
            x="x",
            y="y",
            color="topic",
            size="num_docs",
            title="Main Topic Plot",
        )
        fig.update_layout(clickmode="event+select")
        return fig

    def get_top_words_for_topic(topic_number, model):
        return model.topic_dict.get(topic_number, [])

    @app.callback(
        Output("side-plot", "figure"),
        [Input("main-plot", "clickData"), Input("side-dropdown", "value")],
    )
    def update_side_plot(clickData, side_option):
        if clickData:
            point_index = clickData["points"][0]["pointIndex"]
            selected_topic = point_index

            if side_option == "top_words":
                top_words_data = get_top_words_for_topic(selected_topic, model)
                words, scores = zip(*top_words_data)
                fig = px.bar(
                    x=words, y=scores, title=f"Top Words for Topic {selected_topic}"
                )
                return fig

            elif side_option == "topic_distances":
                distances_data = calculate_distances_to_other_topics(
                    point_index, plot_df, model
                )
                topics, distances, annotations = zip(*distances_data)
                fig = px.bar(
                    x=topics,
                    y=distances,
                    title=f"Distances from Topic {point_index} to Others",
                )
                # Add annotations
                fig.update_layout(
                    annotations=[
                        dict(
                            x=topic,
                            y=distance,
                            text=annotation,
                            showarrow=True,
                            arrowhead=1,
                            ax=0,
                            ay=-40,
                        )
                        for topic, distance, annotation in zip(
                            topics, distances, annotations
                        )
                    ]
                )

                fig.update_traces(
                    hovertemplate="<br>".join(
                        [
                            "Topic ID: %{x}",
                            "Distance: %{y:.2f}",
                            "<extra></extra>",  # Hides the trace name
                        ]
                    )
                )
                return fig

        return go.Figure()

    @app.callback(
        Output("details-panel", "children"), [Input("main-plot", "clickData")]
    )
    def display_details(clickData):
        if clickData:
            point_index = clickData["points"][0]["pointIndex"]
            selected_topic = point_index
            top_words_data = get_top_words_for_topic(selected_topic, model)
            detailed_info = f"Topic {selected_topic}: " + ", ".join(
                [f"{word} ({score:.2f})" for word, score in top_words_data]
            )
            return detailed_info
        return "Click a point for details"

    app.run_server(debug=True, port=port)


def _visualize_topic_model_3d(
    model,
    reduce_first=False,
    reducer="umap",
    port=8050,
    dataset=None,
    encoder_model="paraphrase-MiniLM-L3-v2",
    use_average=True,
):
    """
    Visualize a topic model in 3D space using UMAP, t-SNE, or PCA dimensionality reduction techniques.

    Parameters:
    -----------
    model : object
        Topic modeling model object.
    reduce_first : bool, optional
        Whether to reduce dimensions of embeddings first (default is False).
    reducer : str, optional
        Dimensionality reduction technique to use, one of ['umap', 'tsne', 'pca'] (default is 'umap').
    port : int, optional
        Port number for running the visualization dashboard (default is 8050).

    Returns:
    --------
    None
        The function launches a Dash server to visualize the topic model.
    """
    if not hasattr(model, "embeddings"):
        encoder = SentenceEncodingMixin()
        embeddings = encoder.encode_documents(
            dataset.texts, encoder_model=encoder_model, use_average=use_average
        )
    else:
        embeddings = model.embeddings

    num_docs_per_topic = pd.Series(model.labels).value_counts().sort_index()

    # Extract top words for each topic with importance and format them vertically
    top_words_per_topic = {
        topic: "<br>".join(
            [f"{word} ({importance:.2f})" for word, importance in words[:5]]
        )
        for topic, words in model.topic_dict.items()
    }

    if reducer == "umap":
        reducer = umap.UMAP(n_components=2)
    elif reducer == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, learning_rate=200)
    elif reducer == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("reducer must be in ['umap', 'tnse', 'pca']")

    if reduce_first:
        embeddings = reducer.fit_transform(embeddings)
        topic_data = []
        # Iterate over unique labels and compute mean embedding for each
        for label in np.unique(model.labels):
            # Find embeddings corresponding to the current label
            label_embeddings = embeddings[model.labels == label]
            # Compute mean embedding for the current label
            mean_embedding = np.mean(label_embeddings, axis=0)
            # Store the mean embedding in the dictionary
            topic_data.append(mean_embedding)
    else:
        if hasattr(model, "topic_centroids"):
            topic_data = model.topic_centroids
        else:
            topic_data = []
            # Iterate over unique labels and compute mean embedding for each
            for label in np.unique(model.labels):
                # Find embeddings corresponding to the current label
                label_embeddings = embeddings[model.labels == label]
                # Compute mean embedding for the current label
                mean_embedding = np.mean(label_embeddings, axis=0)
                # Store the mean embedding in the dictionary
                topic_data.append(mean_embedding)

    reducer = umap.UMAP(n_components=3)
    topic_embeddings_3d = reducer.fit_transform(topic_data)

    # Create DataFrame for plotting
    plot_df = pd.DataFrame(topic_embeddings_3d, columns=["x", "y", "z"])
    plot_df["topic"] = list(top_words_per_topic.keys())
    plot_df["num_docs"] = num_docs_per_topic
    plot_df["top_words"] = list(top_words_per_topic.values())

    app = dash.Dash(__name__)

    app.layout = html.Div(
        [
            html.H1("Topic Visualization"),
            dcc.Dropdown(
                id="side-dropdown",
                options=[
                    {"label": "Top Words", "value": "top_words"},
                    {"label": "Topic Distances", "value": "topic_distances"},
                ],
                value="top_words",
            ),
            dcc.Graph(id="main-plot"),
            dcc.Graph(id="side-plot"),
            html.Div(id="details-panel", children="Click a point for details"),
        ]
    )

    @app.callback(Output("main-plot", "figure"), [Input("side-dropdown", "value")])
    def update_main_plot(selected_option):
        fig = px.scatter_3d(
            plot_df,
            x="x",
            y="y",
            z="z",
            color="topic",
            size="num_docs",
            title="Main Topic Plot",
        )
        fig.update_layout(clickmode="event+select")
        return fig

    def get_top_words_for_topic(topic_number, model):
        return model.topics_dict.get(topic_number, [])

    @app.callback(
        Output("side-plot", "figure"),
        [Input("main-plot", "clickData"), Input("side-dropdown", "value")],
    )
    def update_side_plot(clickData, side_option):
        if clickData:
            point_index = clickData["points"][0]["pointNumber"]
            selected_topic = point_index

            if side_option == "top_words":
                top_words_data = get_top_words_for_topic(selected_topic, model)
                words, scores = zip(*top_words_data)
                fig = px.bar(
                    x=words, y=scores, title=f"Top Words for Topic {selected_topic}"
                )
                return fig

            elif side_option == "topic_distances":
                distances_data = calculate_distances_to_other_topics(
                    point_index, plot_df, model
                )
                topics, distances, annotations = zip(*distances_data)
                fig = px.bar(
                    x=topics,
                    y=distances,
                    title=f"Distances from Topic {point_index} to Others",
                )
                # Add annotations
                fig.update_layout(
                    annotations=[
                        dict(
                            x=topic,
                            y=distance,
                            text=annotation,
                            showarrow=True,
                            arrowhead=1,
                            ax=0,
                            ay=-40,
                        )
                        for topic, distance, annotation in zip(
                            topics, distances, annotations
                        )
                    ]
                )

                fig.update_traces(
                    hovertemplate="<br>".join(
                        [
                            "Topic ID: %{x}",
                            "Distance: %{y:.2f}",
                            "<extra></extra>",  # Hides the trace name
                        ]
                    )
                )
                return fig

        return go.Figure()

    @app.callback(
        Output("details-panel", "children"), [Input("main-plot", "clickData")]
    )
    def display_details(clickData):
        if clickData:
            point_index = clickData["points"][0]["pointNumber"]
            selected_topic = point_index
            top_words_data = get_top_words_for_topic(selected_topic, model)
            detailed_info = f"Topic {selected_topic}: " + ", ".join(
                [f"{word} ({score:.2f})" for word, score in top_words_data]
            )
            return detailed_info
        return "Click a point for details"

    app.run_server(debug=True, port=port)


def get_top_tfidf_words_per_document(corpus, n=10):
    """
    Extract top TF-IDF weighted words per document in the corpus.

    Parameters:
    -----------
    corpus : list
        List of documents (strings).
    n : int, optional
        Number of top words to extract per document (default is 10).

    Returns:
    --------
    list of lists
        List of lists containing top TF-IDF weighted words for each document.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    top_words_per_document = []
    for row in X:
        sorted_indices = np.argsort(row.toarray()).flatten()[::-1]
        top_n_indices = sorted_indices[:n]
        top_words = [(feature_names[i], row[0, i]) for i in top_n_indices]
        top_words_per_document.append(top_words)

    return top_words_per_document


def _visualize_topics_2d(
    model,
    reducer="umap",
    port=8050,
    dataset=None,
    encoder_model="paraphrase-MiniLM-L3-v2",
    use_average=True,
    embeddings_path=None,
    embeddings_file_path=None,
):
    """
    Visualize topics in 2D space using UMAP, t-SNE, or PCA dimensionality reduction techniques.

    Parameters:
    -----------
    model : object
        Topic modeling model object.
    reducer : str, optional
        Dimensionality reduction technique to use, one of ['umap', 'tsne', 'pca'] (default is 'umap').
    port : int, optional
        Port number for running the visualization dashboard (default is 8050).

    Returns:
    --------
    None
        The function launches a Dash server to visualize the topic model.
    """
    if not hasattr(model, "embeddings"):
        if dataset.has_embeddings(encoder_model):
            embeddings = dataset.get_embeddings(
                encoder_model,
                embeddings_path,
                embeddings_file_path,
            )
        else:
            encoder = SentenceEncodingMixin()
            embeddings = encoder.encode_documents(
                dataset.texts, encoder_model=encoder_model, use_average=use_average
            )
    else:
        embeddings = model.embeddings
    labels = model.labels
    top_words_per_document = get_top_tfidf_words_per_document(
        model.dataframe["text"])

    # Reduce embeddings to 2D for visualization
    if reducer == "umap":
        reducer = umap.UMAP(n_components=2)
    elif reducer == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, learning_rate=200)
    elif reducer == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("reducer must be in ['umap', 'tnse', 'pca']")
    reduced_embeddings = reducer.fit_transform(embeddings)

    # Prepare DataFrame for scatter plot
    df = pd.DataFrame(reduced_embeddings, columns=["x", "y"])
    df["label"] = labels
    df["top_words"] = [
        "<br>".join([f"{word} ({score:.2f})" for word, score in words])
        for words in top_words_per_document
    ]

    app = dash.Dash(__name__)

    # Unique labels for dropdown with an option to show all
    unique_labels = ["All"] + list(df["label"].unique())

    # App layout
    app.layout = html.Div(
        [
            html.H1("Model Visualization Dashboard"),
            dcc.Dropdown(
                id="topic-dropdown",
                options=[{"label": i, "value": i} for i in unique_labels],
                value="All",
                clearable=False,
            ),
            dcc.Slider(
                id="num-top-words-slider",
                min=1,
                max=25,
                value=5,
                marks={i: str(i) for i in range(1, 11)},
                step=1,
            ),
            dcc.Graph(id="scatter-plot"),
            html.Div(id="info-panel"),
        ]
    )

    # Callback for updating scatter plot
    @app.callback(
        Output("scatter-plot", "figure"),
        [Input("num-top-words-slider", "value"),
         Input("topic-dropdown", "value")],
    )
    def update_plot(num_top_words, selected_topic):
        if selected_topic == "All":
            plot_df = df.copy()
            filtered_top_words = top_words_per_document
        else:
            plot_df = df[df["label"] == selected_topic].copy()
            # Filter top_words_per_document to match the filtered plot_df
            filtered_top_words = [
                words
                for label, words in zip(labels, top_words_per_document)
                if label == selected_topic
            ]

        plot_df["top_words"] = [
            "<br>".join(
                [f"{word} ({score:.2f})" for word,
                 score in words[:num_top_words]]
            )
            for words in filtered_top_words
        ]

        fig = px.scatter(
            plot_df,
            x="x",
            y="y",
            color="label",
            hover_data={"top_words": True},
            labels={"color": "Cluster"},
            title=f"Document Clusters {'(All Topics)' if selected_topic == 'All' else f'for Topic: {selected_topic}'}",
        )
        fig.update_traces(marker=dict(size=8))
        fig.update_layout(width=1000, height=800)
        return fig

    # Callback for info panel
    @app.callback(
        Output("info-panel", "children"), [Input("scatter-plot", "clickData")]
    )
    def display_info(clickData):
        if clickData:
            idx = clickData["points"][0]["pointIndex"]
            return html.P(f"Document ID: {idx}, More info here...")
        return "Click on a point to see more information."

    app.run_server(debug=True, port=port)


def _visualize_topics_3d(
    model,
    reducer="umap",
    port=8050,
    dataset=None,
    encoder_model="paraphrase-MiniLM-L3-v2",
    use_average=True,
    embeddings_path=None,
    embeddings_file_path=None,
):
    """
    Visualize topics in 3D space using UMAP, t-SNE, or PCA dimensionality reduction techniques.

    Parameters:
    -----------
    model : object
        Topic modeling model object.
    reducer : str, optional
        Dimensionality reduction technique to use, one of ['umap', 'tsne', 'pca'] (default is 'umap').
    port : int, optional
        Port number for running the visualization dashboard (default is 8050).

    Returns:
    --------
    None
        The function launches a Dash server to visualize the topic model.
    """
    if not hasattr(model, "embeddings"):
        if dataset.has_embeddings(encoder_model):
            embeddings = dataset.get_embeddings(
                encoder_model,
                embeddings_path,
                embeddings_file_path,
            )
        else:
            encoder = SentenceEncodingMixin()
            embeddings = encoder.encode_documents(
                dataset.texts, encoder_model=encoder_model, use_average=use_average
            )
    else:
        embeddings = model.embeddings
    labels = model.labels
    top_words_per_document = get_top_tfidf_words_per_document(
        model.dataframe["text"])

    # Reduce embeddings to 3D for visualization
    if reducer == "umap":
        reducer = umap.UMAP(n_components=3)
    elif reducer == "tsne":
        reducer = TSNE(n_components=3, perplexity=30, learning_rate=200)
    elif reducer == "pca":
        reducer = PCA(n_components=3)
    else:
        raise ValueError("reducer must be in ['umap', 'tnse', 'pca']")
    reduced_embeddings = reducer.fit_transform(embeddings)

    # Prepare DataFrame for scatter plot
    df = pd.DataFrame(reduced_embeddings, columns=["x", "y", "z"])
    df["label"] = labels
    df["top_words"] = [
        "<br>".join([f"{word} ({score:.2f})" for word, score in words])
        for words in top_words_per_document
    ]

    app = dash.Dash(__name__)

    # Unique labels for dropdown with an option to show all
    unique_labels = ["All"] + list(df["label"].unique())

    # App layout
    app.layout = html.Div(
        [
            html.H1("Model Visualization Dashboard"),
            dcc.Dropdown(
                id="topic-dropdown",
                options=[{"label": i, "value": i} for i in unique_labels],
                value="All",
                clearable=False,
            ),
            dcc.Slider(
                id="num-top-words-slider",
                min=1,
                max=25,
                value=5,
                marks={i: str(i) for i in range(1, 11)},
                step=1,
            ),
            dcc.Graph(id="scatter-plot"),
            html.Div(id="info-panel"),
        ]
    )

    # Callback for updating scatter plot
    @app.callback(
        Output("scatter-plot", "figure"),
        [Input("num-top-words-slider", "value"),
         Input("topic-dropdown", "value")],
    )
    def update_plot(num_top_words, selected_topic):
        if selected_topic == "All":
            plot_df = df.copy()
            filtered_top_words = top_words_per_document
        else:
            plot_df = df[df["label"] == selected_topic].copy()
            # Filter top_words_per_document to match the filtered plot_df
            filtered_top_words = [
                words
                for label, words in zip(labels, top_words_per_document)
                if label == selected_topic
            ]

        plot_df["top_words"] = [
            "<br>".join(
                [f"{word} ({score:.2f})" for word,
                 score in words[:num_top_words]]
            )
            for words in filtered_top_words
        ]

        fig = px.scatter_3d(
            plot_df,
            x="x",
            y="y",
            z="z",
            color="label",
            hover_data={"top_words": True},
            labels={"color": "Cluster"},
            title=f"Document Clusters {'(All Topics)' if selected_topic == 'All' else f'for Topic: {selected_topic}'}",
        )
        fig.update_traces(marker=dict(size=8))
        fig.update_layout(width=1000, height=800)
        return fig

    # Callback for info panel
    @app.callback(
        Output("info-panel", "children"), [Input("scatter-plot", "clickData")]
    )
    def display_info(clickData):
        if clickData:
            idx = clickData["points"][0]["pointNumber"]
            return html.P(f"Document ID: {idx}, More info here...")
        return "Click on a point to see more information."

    app.run_server(debug=True, port=port)
