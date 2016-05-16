import plotly as py
import plotly.graph_objs as go

feature_type = ["LBP","Haralicks","Haralicks and LBP", "SURF", "Haralicks, LBP and SURF"]
scores = [4.8, 9.1, 14.4, 37.3, 47.9]

# Create a trace
trace = go.Scatter(
    x = feature_type,
    y = scores,
    mode = 'lines+markers',
    name = 'lines+markers'
)

data = [trace]

layout = go.Layout(
    title='Feature Set vs Model Performance',
    hovermode='closest',
    xaxis=dict(
        title='Feature Set'
    ),
    yaxis=dict(
        title='Model Performance (%)'
    ),
)

fig = go.Figure(data=data, layout=layout)
py.offline.plot(fig, filename='final_model_results')