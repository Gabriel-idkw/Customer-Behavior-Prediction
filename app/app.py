"""
Customer Segmentation — Interactive Gradio App
================================================
Loads the trained pipeline exported by Notebook 05 (`models/final_unified_pipeline.pkl`)
and lets a user enter a customer's raw RFM profile to get an instant cluster /
persona prediction, with a live radar chart showing where they land relative
to the two discovered personas.

Run locally (from the `app/` directory, after Notebook 05 has been run in Colab
and `models/final_unified_pipeline.pkl` exists):

    pip install -r requirements.txt
    python app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import gradio as gr

from rfm_pipeline import RFMFeatureTransformer, FEATURE_ORDER  # noqa: F401 (needed to unpickle the pipeline)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'final_unified_pipeline.pkl'
PERSONA_PATH = PROJECT_ROOT / 'models' / 'persona_labels.pkl'
SUMMARY_PATH = PROJECT_ROOT / 'metrics' / 'final_comparison_summary.json'

pipeline = joblib.load(MODEL_PATH)
persona_labels = joblib.load(PERSONA_PATH)

with open(SUMMARY_PATH) as f:
    summary = json.load(f)
winning_algo = summary['winning_algorithm']
persona_rows = {row['cluster']: row for row in summary['persona_table']}
raw_cols = ['Recency', 'Frequency', 'Monetary', 'AOV', 'Return_Rate_Pct']

# Precompute normalization bounds across personas so the radar chart is legible
persona_df = pd.DataFrame(persona_rows.values()).set_index('cluster')[raw_cols]
persona_min = persona_df.min()
persona_max = persona_df.max()


def normalize(row):
    return ((row - persona_min) / (persona_max - persona_min + 1e-9)).clip(0, 1)


PERSONA_COLORS = ['#4C9AFF', '#F87171', '#34D399', '#FBBF24']


def make_radar(customer_norm, predicted_cluster):
    categories = raw_cols
    theta = categories + [categories[0]]

    fig = go.Figure()

    for i, (cid, row) in enumerate(persona_df.iterrows()):
        norm_row = normalize(row)
        r = norm_row.tolist() + [norm_row.tolist()[0]]
        is_pred = (cid == predicted_cluster)
        color = PERSONA_COLORS[i % len(PERSONA_COLORS)]
        fig.add_trace(go.Scatterpolar(
            r=r, theta=theta,
            name=persona_rows[cid]['Persona'],
            fill='toself',
            line=dict(color=color, width=3 if is_pred else 1, dash='solid' if is_pred else 'dot'),
            opacity=0.85 if is_pred else 0.35,
        ))

    cust_r = customer_norm.tolist() + [customer_norm.tolist()[0]]
    fig.add_trace(go.Scatterpolar(
        r=cust_r, theta=theta,
        name='This customer',
        line=dict(color='white', width=3),
        marker=dict(size=9, color='white', symbol='circle'),
    ))

    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor='rgba(255,255,255,0.15)'),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.15)'),
        ),
        showlegend=True,
        legend=dict(orientation='h', yanchor='top', y=-0.1, xanchor='center', x=0.5, font=dict(size=11)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e5e5e5'),
        margin=dict(l=60, r=60, t=30, b=60),
        autosize=True,
        height=440,
    )
    return fig


def predict_customer(recency, frequency, monetary, aov, return_rate_pct):
    x = pd.DataFrame([{
        'Recency': recency, 'Frequency': frequency, 'Monetary': monetary,
        'AOV': aov, 'Return_Rate_Pct': return_rate_pct,
    }])
    cluster = int(pipeline.predict(x)[0])
    persona = persona_rows.get(cluster, {}).get('Persona', f'Cluster {cluster}')

    result_md = f"""
### Predicted Segment: **{persona}**

| Feature | Customer Value | Segment Average |
|---|---|---|
| Recency (days since last order) | {recency:.0f} | {persona_rows[cluster]['Recency']:.0f} |
| Frequency (total orders) | {frequency:.0f} | {persona_rows[cluster]['Frequency']:.1f} |
| Monetary (total spend, $) | ${monetary:,.0f} | ${persona_rows[cluster]['Monetary']:,.0f} |
| Avg. Order Value ($) | ${aov:,.0f} | ${persona_rows[cluster]['AOV']:,.0f} |
| Return Rate (%) | {return_rate_pct:.1f}% | {persona_rows[cluster]['Return_Rate_Pct']:.1f}% |

*Model: {winning_algo} (selected by Silhouette Score in Notebook 05's model comparison)*
"""
    customer_row = pd.Series({'Recency': recency, 'Frequency': frequency, 'Monetary': monetary,
                               'AOV': aov, 'Return_Rate_Pct': return_rate_pct})
    fig = make_radar(normalize(customer_row), cluster)
    return result_md, fig


PRESETS = {
    "VIP example": (3, 24, 28000, 1150, 3),
    "At-risk example": (500, 2, 3000, 1600, 10),
    "Typical example": (187, 5, 13000, 2700, 40),
    "Recently Active example": (30, 1, 1000, 1000, 0),
}


def load_preset(name):
    return PRESETS[name]


with gr.Blocks(title="Customer Segmentation Explorer") as demo:
    gr.Markdown(f"""
    # Customer Segmentation Explorer
    Enter a customer's RFM profile below to see which behavioral segment they fall into

    """)

    with gr.Row():
        with gr.Column(scale=1):
            preset = gr.Dropdown(choices=list(PRESETS.keys()), label="Load an example", value=None)
            recency = gr.Slider(0, 810, value=187, step=1, label="Recency (days since last purchase)")
            frequency = gr.Slider(1, 30, value=5, step=1, label="Frequency (total orders)")
            monetary = gr.Slider(100, 55000, value=13000, step=100, label="Monetary (total $ spent)")
            aov = gr.Slider(500, 5000, value=2700, step=50, label="Average Order Value ($)")
            return_rate = gr.Slider(0, 100, value=40, step=1, label="Return Rate (%)")
            predict_btn = gr.Button("Predict Segment", variant="primary")

        with gr.Column(scale=1):
            output_md = gr.Markdown()
            output_plot = gr.Plot()

    inputs = [recency, frequency, monetary, aov, return_rate]
    outputs = [output_md, output_plot]

    # Live prediction: fires the moment you drop a slider, no button click needed.
    for s in inputs:
        s.release(fn=predict_customer, inputs=inputs, outputs=outputs)

    # Loading a preset should also predict immediately, and the button still works
    # for anyone who prefers clicking (or is on a device where release doesn't fire).
    preset.change(fn=load_preset, inputs=preset, outputs=inputs).then(
        fn=predict_customer, inputs=inputs, outputs=outputs)
    predict_btn.click(fn=predict_customer, inputs=inputs, outputs=outputs)
    demo.load(fn=predict_customer, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    demo.launch()
