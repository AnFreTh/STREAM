import numpy as np
import plotly.graph_objects as go
import torch
from dash import Dash, Input, Output, dcc, html


def plot_with_plotly(feature_nn, feature_name, feature_values, y_min, y_max):
    with torch.no_grad():
        # Determine if the feature is likely continuous or categorical/binary
        unique_values = np.unique(feature_values)
        if len(unique_values) > 10:  # Arbitrary threshold, adjust based on your data
            # Treat as continuous
            # Generate simulated 'x' values within the range of observed values for smoother plotting
            x_simulated = np.linspace(
                feature_values.min(), feature_values.max(), 250)
        else:
            # Treat as categorical/binary, use unique values directly
            x_simulated = unique_values

        # Convert simulated 'x' values to a Tensor
        x_tensor = torch.Tensor(x_simulated).unsqueeze(1)

        # Make predictions using the simulated 'x' values
        feature_nn.eval()
        with torch.no_grad():
            predictions = feature_nn.forward(x_tensor).numpy().flatten()

        # Create a Plotly figure
        fig = go.Figure()

        # Plot the simulated 'x' values against the predicted 'y' values
        fig.add_trace(
            go.Scatter(
                x=x_simulated,
                y=predictions,
                mode="lines",
                name=feature_name,
                line=dict(color="red"),
            )
        )

        # Update layout with feature name as y-axis label and set y-axis range
        fig.update_layout(
            yaxis_title=feature_name,
            title=f"Feature: {feature_name}",
            xaxis_title="Input Range",
            # yaxis=dict(range=[y_min, y_max]),
        )

        return fig


def plot_downstream_model(downstream_model):
    app = Dash(__name__)

    feature_names = downstream_model.combined_data.columns[
        :-1
    ].tolist()  # Exclude the target column if present

    y_min = np.min(
        downstream_model.combined_data[downstream_model.combined_data.columns[-1]]
    )
    y_max = np.max(
        downstream_model.combined_data[downstream_model.combined_data.columns[-1]]
    )

    app.layout = html.Div(
        [
            html.H1("Feature-specific Neural Network Functions"),
            dcc.Dropdown(
                id="feature-dropdown",
                options=[{"label": name, "value": name}
                         for name in feature_names],
                value=feature_names[0],  # Default value
            ),
            dcc.Graph(id="feature-graph"),
        ]
    )

    @app.callback(
        Output("feature-graph", "figure"), [Input("feature-dropdown", "value")]
    )
    def update_graph(selected_feature):
        feature_index = feature_names.index(selected_feature)
        feature_nn = downstream_model.model.feature_nns[feature_index]

        # Retrieve the true data values for the selected feature
        feature_values = downstream_model.combined_data[selected_feature]

        # Pass the true data values and global y-axis range to the plot function
        return plot_with_plotly(
            feature_nn, selected_feature, feature_values, y_min, y_max
        )

    app.run_server(debug=True)
