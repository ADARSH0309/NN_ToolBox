import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from backprop import train_network, predict, DEFAULT_LR, DEFAULT_EPOCHS
from perceptron import train_perceptron, predict_perceptron, DEFAULT_LR as P_DEFAULT_LR, DEFAULT_EPOCHS as P_DEFAULT_EPOCHS
from rnn import train_rnn, DEFAULT_HIDDEN as R_DEFAULT_HIDDEN, DEFAULT_LR as R_DEFAULT_LR, DEFAULT_EPOCHS as R_DEFAULT_EPOCHS
from mse import train_mse_single, predict_single, train_mse_dual, predict_dual, DEFAULT_LR as M_DEFAULT_LR, DEFAULT_EPOCHS as M_DEFAULT_EPOCHS
from visualizations import (
    plot_decision_boundary, plot_confidence_heatmap, plot_weight_heatmap_mlp,
    plot_confusion_matrix, plot_activation_distribution, plot_loss_curve,
    plot_regression_line, plot_regression_3d, plot_perceptron_boundary,
    plot_residual, plot_sentiment_distribution, plot_loss_accuracy, plot_word_frequency
)


def backprop_diagram():
    return """
    <html>
    <head>
        <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            body { margin: 0; background: transparent; }
            #network { width: 100%; height: 500px; border: 1px solid #444; border-radius: 8px; background: #1e1e1e; }
        </style>
    </head>
    <body>
        <div id="network"></div>
        <script>
            var nodes = new vis.DataSet([
                {id: 1, label: 'x1\\n(Input 1)', color: '#4CAF50', font: {color: 'white'}, group: 'input', x: -300, y: -100},
                {id: 2, label: 'x2\\n(Input 2)', color: '#4CAF50', font: {color: 'white'}, group: 'input', x: -300, y: 100},
                {id: 3, label: 'h1\\n(Hidden)', color: '#2196F3', font: {color: 'white'}, group: 'hidden', x: 0, y: -100},
                {id: 4, label: 'h2\\n(Hidden)', color: '#2196F3', font: {color: 'white'}, group: 'hidden', x: 0, y: 100},
                {id: 5, label: 'o\\n(Output)', color: '#FF5722', font: {color: 'white'}, group: 'output', x: 300, y: 0},
                {id: 6, label: 'bh1', color: '#9E9E9E', font: {color: 'white'}, shape: 'box', x: 0, y: -250},
                {id: 7, label: 'bh2', color: '#9E9E9E', font: {color: 'white'}, shape: 'box', x: 0, y: 250},
                {id: 8, label: 'bo', color: '#9E9E9E', font: {color: 'white'}, shape: 'box', x: 300, y: -150}
            ]);

            var edges = new vis.DataSet([
                {from: 1, to: 3, label: 'w1', color: '#aaa', font: {color: '#fff', strokeWidth: 0}},
                {from: 1, to: 4, label: 'w2', color: '#aaa', font: {color: '#fff', strokeWidth: 0}},
                {from: 2, to: 3, label: 'w3', color: '#aaa', font: {color: '#fff', strokeWidth: 0}},
                {from: 2, to: 4, label: 'w4', color: '#aaa', font: {color: '#fff', strokeWidth: 0}},
                {from: 3, to: 5, label: 'w5', color: '#FF9800', font: {color: '#fff', strokeWidth: 0}, width: 2},
                {from: 4, to: 5, label: 'w6', color: '#FF9800', font: {color: '#fff', strokeWidth: 0}, width: 2},
                {from: 6, to: 3, color: '#666', dashes: true},
                {from: 7, to: 4, color: '#666', dashes: true},
                {from: 8, to: 5, color: '#666', dashes: true}
            ]);

            var container = document.getElementById('network');
            var data = {nodes: nodes, edges: edges};
            var options = {
                nodes: { shape: 'circle', size: 35, font: {size: 12} },
                edges: { arrows: 'to', smooth: {type: 'curvedCW', roundness: 0.1} },
                physics: { enabled: false },
                interaction: { dragNodes: true, dragView: true, zoomView: true }
            };
            var network = new vis.Network(container, data, options);
        </script>
    </body>
    </html>
    """


def perceptron_diagram():
    return """
    <html>
    <head>
        <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            body { margin: 0; background: transparent; }
            #network { width: 100%; height: 500px; border: 1px solid #444; border-radius: 8px; background: #1e1e1e; }
        </style>
    </head>
    <body>
        <div id="network"></div>
        <script>
            var nodes = new vis.DataSet([
                {id: 1, label: 'x1\\n(Input 1)', color: '#4CAF50', font: {color: 'white'}, x: -300, y: -100},
                {id: 2, label: 'x2\\n(Input 2)', color: '#4CAF50', font: {color: 'white'}, x: -300, y: 100},
                {id: 3, label: 'bias\\n(1)', color: '#9E9E9E', font: {color: 'white'}, x: -300, y: 0},
                {id: 4, label: 'Σ\\n(Sum)', color: '#2196F3', font: {color: 'white'}, size: 40, x: 0, y: 0},
                {id: 5, label: 'Step\\nFunction', color: '#FF9800', font: {color: 'white'}, shape: 'box', x: 200, y: 0},
                {id: 6, label: 'y\\n(Output)', color: '#E91E63', font: {color: 'white'}, x: 400, y: 0}
            ]);

            var edges = new vis.DataSet([
                {from: 1, to: 4, label: 'w1', color: '#aaa', font: {color: '#fff', strokeWidth: 0}},
                {from: 2, to: 4, label: 'w2', color: '#aaa', font: {color: '#fff', strokeWidth: 0}},
                {from: 3, to: 4, label: 'b', color: '#666', font: {color: '#fff', strokeWidth: 0}, dashes: true},
                {from: 4, to: 5, color: '#FF9800', width: 2},
                {from: 5, to: 6, label: '0 or 1', color: '#E91E63', font: {color: '#fff', strokeWidth: 0}, width: 2}
            ]);

            var container = document.getElementById('network');
            var data = {nodes: nodes, edges: edges};
            var options = {
                nodes: { shape: 'circle', size: 35, font: {size: 12} },
                edges: { arrows: 'to', smooth: {type: 'curvedCW', roundness: 0.1} },
                physics: { enabled: false },
                interaction: { dragNodes: true, dragView: true, zoomView: true }
            };
            var network = new vis.Network(container, data, options);
        </script>
    </body>
    </html>
    """

def rnn_diagram():
    return """
    <html>
    <head>
        <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            body { margin: 0; background: transparent; }
            #network { width: 100%; height: 500px; border: 1px solid #444; border-radius: 8px; background: #1e1e1e; }
        </style>
    </head>
    <body>
        <div id="network"></div>
        <script>
            var nodes = new vis.DataSet([
                {id: 1,  label: 'x₁\\n(word 1)', color: '#4CAF50', font: {color: 'white'}, x: -400, y: 0},
                {id: 2,  label: 'x₂\\n(word 2)', color: '#4CAF50', font: {color: 'white'}, x: -200, y: 0},
                {id: 3,  label: 'x₃\\n(word 3)', color: '#4CAF50', font: {color: 'white'}, x: 0, y: 0},
                {id: 4,  label: '...', color: '#666', font: {color: 'white'}, shape: 'text', x: 150, y: 0},
                {id: 5,  label: 'xₜ\\n(last)', color: '#4CAF50', font: {color: 'white'}, x: 300, y: 0},
                {id: 11, label: 'h₁', color: '#2196F3', font: {color: 'white'}, x: -400, y: -150},
                {id: 12, label: 'h₂', color: '#2196F3', font: {color: 'white'}, x: -200, y: -150},
                {id: 13, label: 'h₃', color: '#2196F3', font: {color: 'white'}, x: 0, y: -150},
                {id: 15, label: 'hₜ', color: '#2196F3', font: {color: 'white'}, x: 300, y: -150},
                {id: 20, label: 'σ(y)\\nOutput', color: '#FF5722', font: {color: 'white'}, size: 40, x: 300, y: -300},
                {id: 30, label: 'Embed', color: '#9C27B0', font: {color: 'white'}, shape: 'box', x: -500, y: 0}
            ]);
            var edges = new vis.DataSet([
                {from: 1, to: 11, label: 'Wₓₕ', color: '#aaa', font: {color:'#fff',strokeWidth:0}},
                {from: 2, to: 12, label: 'Wₓₕ', color: '#aaa', font: {color:'#fff',strokeWidth:0}},
                {from: 3, to: 13, label: 'Wₓₕ', color: '#aaa', font: {color:'#fff',strokeWidth:0}},
                {from: 5, to: 15, label: 'Wₓₕ', color: '#aaa', font: {color:'#fff',strokeWidth:0}},
                {from: 11, to: 12, label: 'Wₕₕ', color: '#FF9800', font: {color:'#fff',strokeWidth:0}, width: 2},
                {from: 12, to: 13, label: 'Wₕₕ', color: '#FF9800', font: {color:'#fff',strokeWidth:0}, width: 2},
                {from: 13, to: 15, label: 'Wₕₕ', color: '#FF9800', font: {color:'#fff',strokeWidth:0}, width: 2, dashes: true},
                {from: 15, to: 20, label: 'Wₕᵧ', color: '#E91E63', font: {color:'#fff',strokeWidth:0}, width: 2},
                {from: 30, to: 1, color: '#9C27B0', dashes: true}
            ]);
            var container = document.getElementById('network');
            var data = {nodes: nodes, edges: edges};
            var options = {
                nodes: { shape: 'circle', size: 35, font: {size: 12} },
                edges: { arrows: 'to', smooth: {type: 'curvedCW', roundness: 0.1} },
                physics: { enabled: false },
                interaction: { dragNodes: true, dragView: true, zoomView: true }
            };
            var network = new vis.Network(container, data, options);
        </script>
    </body>
    </html>
    """


def mse_diagram():
    return """
    <html>
    <head>
        <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style>
            body { margin: 0; background: transparent; }
            #network { width: 100%; height: 500px; border: 1px solid #444; border-radius: 8px; background: #1e1e1e; }
        </style>
    </head>
    <body>
        <div id="network"></div>
        <script>
            var nodes = new vis.DataSet([
                {id: 1, label: 'x₁\\n(Feature 1)', color: '#4CAF50', font: {color: 'white'}, x: -300, y: -100},
                {id: 2, label: 'x₂\\n(Feature 2)', color: '#4CAF50', font: {color: 'white'}, x: -300, y: 100},
                {id: 3, label: 'Σ\\n(Weighted Sum)', color: '#2196F3', font: {color: 'white'}, size: 45, x: 50, y: 0},
                {id: 4, label: 'ŷ\\n(Prediction)', color: '#FF9800', font: {color: 'white'}, x: 300, y: 0},
                {id: 5, label: 'MSE\\nLoss', color: '#F44336', font: {color: 'white'}, shape: 'box', x: 500, y: 0},
                {id: 6, label: 'y\\n(True)', color: '#9C27B0', font: {color: 'white'}, x: 500, y: 150},
                {id: 7, label: 'bias', color: '#9E9E9E', font: {color: 'white'}, shape: 'box', x: 50, y: 200}
            ]);
            var edges = new vis.DataSet([
                {from: 1, to: 3, label: 'w₁', color: '#aaa', font: {color:'#fff',strokeWidth:0}},
                {from: 2, to: 3, label: 'w₂', color: '#aaa', font: {color:'#fff',strokeWidth:0}},
                {from: 7, to: 3, label: 'b', color: '#666', font: {color:'#fff',strokeWidth:0}, dashes: true},
                {from: 3, to: 4, color: '#FF9800', width: 2},
                {from: 4, to: 5, label: '(ŷ-y)²', color: '#F44336', font: {color:'#fff',strokeWidth:0}, width: 2},
                {from: 6, to: 5, color: '#9C27B0', dashes: true}
            ]);
            var container = document.getElementById('network');
            var data = {nodes: nodes, edges: edges};
            var options = {
                nodes: { shape: 'circle', size: 35, font: {size: 12} },
                edges: { arrows: 'to', smooth: {type: 'curvedCW', roundness: 0.1} },
                physics: { enabled: false },
                interaction: { dragNodes: true, dragView: true, zoomView: true }
            };
            var network = new vis.Network(container, data, options);
        </script>
    </body>
    </html>
    """


st.set_page_config(page_title="Neural Network Toolbox", layout="wide")

# ── Theme Toggle ─────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

theme = st.sidebar.toggle("Light Mode", value=(st.session_state.theme == "Light"), key="theme_toggle")
st.session_state.theme = "Light" if theme else "Dark"

if st.session_state.theme == "Light":
    st.markdown("""<style>
        :root {
            --bg: #ffffff; --text: #111111; --card: #f8f9fa; --border: #dee2e6;
        }
        .stApp { background-color: var(--bg); color: var(--text); }
        .stMarkdown, .stText, h1, h2, h3, p, span, label { color: var(--text) !important; }
        .stDataFrame { background: var(--card); }
        div[data-testid="stMetric"] { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
        div[data-testid="stMetricValue"] { color: var(--text) !important; }
        section[data-testid="stSidebar"] { background: #f0f2f6; }
        section[data-testid="stSidebar"] * { color: #111 !important; }
        .stTabs [data-baseweb="tab"] { color: var(--text) !important; }
    </style>""", unsafe_allow_html=True)
else:
    st.markdown("""<style>
        :root {
            --bg: #0e1117; --text: #fafafa; --card: #1a1d24; --border: #333;
        }
        .stApp { background-color: var(--bg); color: var(--text); }
        div[data-testid="stMetric"] { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
    </style>""", unsafe_allow_html=True)

st.title("Neural Network Toolbox")

# Session state
if 'trained' not in st.session_state:
    st.session_state.trained = False
    st.session_state.weights = None
    st.session_state.loss_history = None
    st.session_state.model_type = None

_t = st.session_state.theme  # shorthand for passing theme to visualizations

# Sidebar - Model Selection
st.sidebar.divider()
st.sidebar.header("Select Model")
st.markdown("""<style>div[data-baseweb="select"] input {caret-color: transparent !important;}</style>""", unsafe_allow_html=True)
model_type = st.sidebar.selectbox(
    "Choose Neural Network Type",
    ["MLP (Multi-Layer Perceptron)", "Backpropagation", "Perceptron", "RNN (Sentiment Analysis)", "MSE Loss (Linear Regression)", "Loss Functions Explained"]
)

st.sidebar.divider()

# ============================================
# MLP (Multi-Layer Perceptron) — Architecture
# ============================================
if model_type == "MLP (Multi-Layer Perceptron)":
    st.sidebar.header("MLP Parameters")
    st.sidebar.subheader("Initial Weights")
    st.sidebar.info("Random (generated on each training run)")
    b_lr = st.sidebar.number_input("Learning Rate", min_value=0.00001, max_value=1.0, value=DEFAULT_LR, step=0.0001, format="%.5f")
    b_epochs = st.sidebar.number_input("Epochs", min_value=10, max_value=5000, value=DEFAULT_EPOCHS, step=50)

    st.header("MLP — Multi-Layer Perceptron")
    st.caption("A feedforward neural network: Input Layer → Hidden Layer → Output Layer")

    tab1, tab2, tab3, tab4 = st.tabs(["Upload & Train", "Predict", "Visualizations", "Architecture"])

    with tab1:
        st.subheader("Step 1: Upload CSV File")
        st.info("CSV format: First 2 columns = features, Last column = label (0 or 1)")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="mlp_upload")
        use_sample = st.checkbox("Use sample data instead", key="mlp_sample")

        if use_sample:
            df = pd.read_csv("sample_data.csv")
            st.write("Sample Data (Student Performance):")
            st.dataframe(df, use_container_width=True)
            X = df.iloc[:, :2].values.tolist()
            Y = df.iloc[:, -1].values.tolist()
            st.session_state.col_names = [df.columns[0], df.columns[1]]
        elif uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:")
            st.dataframe(df, use_container_width=True)
            X = df.iloc[:, :2].values.tolist()
            Y = df.iloc[:, -1].values.tolist()
            st.session_state.col_names = [df.columns[0], df.columns[1]]
        else:
            X = None
            Y = None

        st.subheader("Step 2: Train MLP")

        if X is not None and Y is not None:
            if st.button("Train MLP", type="primary", key="mlp_train"):
                with st.spinner("Training..."):
                    weights, loss_history, init_weights = train_network(X, Y, l_rate=b_lr, n_epochs=b_epochs)
                    st.session_state.trained = True
                    st.session_state.weights = weights
                    st.session_state.loss_history = loss_history
                    st.session_state.model_type = "mlp"
                    st.session_state.mlp_X = X
                    st.session_state.mlp_Y = Y

                st.success("Training Complete!")

                iw1, iw2, iw3, iw4, iw5, iw6, _, _, _ = init_weights
                st.subheader("Initial Random Weights")
                icol1, icol2 = st.columns(2)
                with icol1:
                    st.write(f"w1: {iw1}  w3: {iw3}  w5: {iw5}")
                with icol2:
                    st.write(f"w2: {iw2}  w4: {iw4}  w6: {iw6}")

                w1, w2, w3, w4, w5, w6, bh1, bh2, bo = weights
                st.subheader("Trained Weights")
                wcol1, wcol2, wcol3 = st.columns(3)
                with wcol1:
                    st.write(f"w1: {w1:.6f}")
                    st.write(f"w2: {w2:.6f}")
                    st.write(f"w3: {w3:.6f}")
                with wcol2:
                    st.write(f"w4: {w4:.6f}")
                    st.write(f"w5: {w5:.6f}")
                    st.write(f"w6: {w6:.6f}")
                with wcol3:
                    st.write(f"bh1: {bh1:.6f}")
                    st.write(f"bh2: {bh2:.6f}")
                    st.write(f"bo: {bo:.6f}")

                st.subheader("Training Loss Over Epochs")
                loss_df = pd.DataFrame({'Epoch': range(1, len(loss_history) + 1), 'Loss': loss_history})
                st.line_chart(loss_df.set_index('Epoch'))
        else:
            st.warning("Please upload a CSV file or use sample data to train.")

    with tab2:
        st.subheader("Predict Output")

        if st.session_state.trained and st.session_state.model_type == "mlp":
            st.success("Model is trained and ready!")
            col_names = st.session_state.get('col_names', ['Feature 1', 'Feature 2'])

            pcol1, pcol2 = st.columns(2)
            with pcol1:
                pred_x1 = st.number_input(col_names[0], value=5.0, key="mlp_x1")
            with pcol2:
                pred_x2 = st.number_input(col_names[1], value=75.0, key="mlp_x2")

            if st.button("Predict", type="primary", key="mlp_predict"):
                o, predicted_class = predict(pred_x1, pred_x2, st.session_state.weights)

                st.subheader("Prediction Result")
                st.metric("Raw Output (o)", f"{o:.6f}")
                st.metric("Predicted Class", "Pass" if predicted_class == 1 else "Fail")
                st.progress(o)
                st.caption(f"Probability of Pass: {o*100:.2f}%")
        else:
            st.warning("Please train the MLP first.")

    with tab3:
        st.subheader("Visualizations")
        if st.session_state.get("trained") and st.session_state.get("model_type") == "mlp":
            viz_type = st.selectbox("Select Visualization", [
                "Decision Boundary", "Confidence Heatmap", "Weight Heatmap",
                "Confusion Matrix", "Activation Distribution", "Loss Curve"
            ], key="mlp_viz")

            weights = st.session_state.weights
            X_data = st.session_state.get("bp_X") or st.session_state.get("mlp_X")
            Y_data = st.session_state.get("bp_Y") or st.session_state.get("mlp_Y")

            if viz_type == "Decision Boundary" and X_data:
                fig = plot_decision_boundary(X_data, Y_data, lambda x1, x2: predict(x1, x2, weights), "MLP Decision Boundary", theme=_t)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Green region = Class 1, Red region = Class 0. The boundary shows where the model switches its prediction.")

            elif viz_type == "Confidence Heatmap" and X_data:
                fig = plot_confidence_heatmap(X_data, Y_data, lambda x1, x2: predict(x1, x2, weights), "MLP Confidence Map", theme=_t)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Brighter green = high confidence for Class 1, Brighter red = high confidence for Class 0. Dashed line = decision boundary.")

            elif viz_type == "Weight Heatmap":
                fig = plot_weight_heatmap_mlp(weights, "MLP Trained Weight Heatmap", theme=_t)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Blue = positive weight, Red = negative weight. Stronger color = larger magnitude.")

            elif viz_type == "Confusion Matrix" and X_data:
                y_pred = [predict(x[0], x[1], weights)[1] for x in X_data]
                fig, metrics = plot_confusion_matrix(Y_data, y_pred, "MLP Confusion Matrix", theme=_t)
                st.plotly_chart(fig, use_container_width=True)
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
                mcol2.metric("Precision", f"{metrics['precision']*100:.1f}%")
                mcol3.metric("Recall", f"{metrics['recall']*100:.1f}%")
                mcol4.metric("F1 Score", f"{metrics['f1']*100:.1f}%")

            elif viz_type == "Activation Distribution" and X_data:
                fig = plot_activation_distribution(X_data, weights, "MLP Neuron Activations", theme=_t)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Shows how each neuron responds across all data points. Ideally, outputs should cluster near 0 or 1.")

            elif viz_type == "Loss Curve" and st.session_state.get("loss_history"):
                fig = plot_loss_curve(st.session_state.loss_history, "MLP Training Loss", theme=_t)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Lower loss = better fit. The curve should decrease and flatten over time.")
        else:
            st.warning("Train the MLP first to see visualizations.")

    with tab4:
        st.subheader("MLP Network Architecture")
        st.caption("Drag nodes to move, scroll to zoom, drag background to pan")
        components.html(backprop_diagram(), height=520)

        st.divider()
        st.subheader("What is an MLP?")
        st.markdown("""
An **MLP (Multi-Layer Perceptron)** is a feedforward neural network with:
- **Input Layer** — receives raw features (e.g. study hours, attendance)
- **Hidden Layer(s)** — learns non-linear patterns using weighted sums + activation functions
- **Output Layer** — produces the final prediction

It is the **architecture** (structure) of the network — how neurons are connected in layers.
""")

        st.divider()
        st.subheader("Step 1: Input Layer → Hidden Layer")
        st.markdown("Each hidden neuron computes a **weighted sum** of all inputs + bias:")
        st.latex(r"z_{h1} = x_1 \cdot w_1 + x_2 \cdot w_3 + b_{h1}")
        st.latex(r"z_{h2} = x_1 \cdot w_2 + x_2 \cdot w_4 + b_{h2}")
        st.code("""zh1 = x1 * w1 + x2 * w3 + bh1
zh2 = x1 * w2 + x2 * w4 + bh2""", language="python")

        st.divider()
        st.subheader("Step 2: Activation Function (Sigmoid)")
        st.markdown("Applies **non-linearity** so the network can learn complex patterns:")
        st.latex(r"h_1 = \sigma(z_{h1}) = \frac{1}{1 + e^{-z_{h1}}}")
        st.latex(r"h_2 = \sigma(z_{h2}) = \frac{1}{1 + e^{-z_{h2}}}")
        st.code("""h1 = 1 / (1 + e^(-zh1))
h2 = 1 / (1 + e^(-zh2))""", language="python")

        st.divider()
        st.subheader("Step 3: Hidden Layer → Output Layer")
        st.markdown("The output neuron combines hidden outputs:")
        st.latex(r"z_o = h_1 \cdot w_5 + h_2 \cdot w_6 + b_o")
        st.latex(r"o = \sigma(z_o) = \frac{1}{1 + e^{-z_o}}")
        st.code("""zo = h1 * w5 + h2 * w6 + bo
o = 1 / (1 + e^(-zo))  # final prediction (0 to 1)""", language="python")

        st.divider()
        st.subheader("MLP vs Backpropagation")
        st.markdown("""
| | MLP | Backpropagation |
|---|---|---|
| **Type** | Network **architecture** | Training **algorithm** |
| **What it does** | Defines layers & connections | Computes gradients & updates weights |
| **Analogy** | The car (structure) | The engine (learning mechanism) |

> Select **"Backpropagation"** from the sidebar to see how the training algorithm works step by step.
""")

# ============================================
# BACKPROPAGATION — Training Algorithm
# ============================================
elif model_type == "Backpropagation":
    st.sidebar.header("Backprop Parameters")
    st.sidebar.subheader("Initial Weights")
    st.sidebar.info("Random (generated on each training run)")
    b_lr = st.sidebar.number_input("Learning Rate", min_value=0.00001, max_value=1.0, value=DEFAULT_LR, step=0.0001, format="%.5f", key="bp_lr")
    b_epochs = st.sidebar.number_input("Epochs", min_value=10, max_value=5000, value=DEFAULT_EPOCHS, step=50, key="bp_epochs")

    st.header("Backpropagation — Training Algorithm")
    st.caption("The algorithm that teaches neural networks by propagating errors backward")

    tab1, tab2, tab3, tab4 = st.tabs(["Upload & Train", "Step-by-Step Trace", "Visualizations", "Algorithm"])

    with tab1:
        st.subheader("Step 1: Upload CSV File")
        st.info("CSV format: First 2 columns = features, Last column = label (0 or 1)")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="bp_upload")
        use_sample = st.checkbox("Use sample data instead", key="bp_sample")

        if use_sample:
            df = pd.read_csv("sample_data.csv")
            st.write("Sample Data (Student Performance):")
            st.dataframe(df, use_container_width=True)
            X = df.iloc[:, :2].values.tolist()
            Y = df.iloc[:, -1].values.tolist()
            st.session_state.col_names = [df.columns[0], df.columns[1]]
        elif uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:")
            st.dataframe(df, use_container_width=True)
            X = df.iloc[:, :2].values.tolist()
            Y = df.iloc[:, -1].values.tolist()
            st.session_state.col_names = [df.columns[0], df.columns[1]]
        else:
            X = None
            Y = None

        st.subheader("Step 2: Train with Backpropagation")

        if X is not None and Y is not None:
            if st.button("Train", type="primary", key="bp_train"):
                with st.spinner("Training with backpropagation..."):
                    weights, loss_history, init_weights = train_network(X, Y, l_rate=b_lr, n_epochs=b_epochs)
                    st.session_state.trained = True
                    st.session_state.weights = weights
                    st.session_state.loss_history = loss_history
                    st.session_state.model_type = "backprop"
                    st.session_state.bp_init_weights = init_weights
                    st.session_state.bp_X = X
                    st.session_state.bp_Y = Y

                st.success("Training Complete!")

                iw1, iw2, iw3, iw4, iw5, iw6, _, _, _ = init_weights
                st.subheader("Initial → Trained Weights")
                w1, w2, w3, w4, w5, w6, bh1, bh2, bo = weights
                weight_df = pd.DataFrame({
                    "Weight": ["w1", "w2", "w3", "w4", "w5", "w6", "bh1", "bh2", "bo"],
                    "Initial": [iw1, iw2, iw3, iw4, iw5, iw6, 0.0, 0.0, 0.0],
                    "Trained": [w1, w2, w3, w4, w5, w6, bh1, bh2, bo]
                })
                st.dataframe(weight_df, use_container_width=True)

                st.subheader("Loss Curve (Error Reduction Over Time)")
                loss_df = pd.DataFrame({'Epoch': range(1, len(loss_history) + 1), 'Total Error': loss_history})
                st.line_chart(loss_df.set_index('Epoch'))
                st.caption("Backpropagation minimizes this error by adjusting weights after each sample.")
        else:
            st.warning("Please upload a CSV file or use sample data to train.")

    with tab2:
        st.subheader("Step-by-Step Trace (1 Sample)")
        st.markdown("See exactly how backpropagation processes **one data point**:")

        if st.session_state.get("model_type") == "backprop" and st.session_state.get("trained"):
            weights = st.session_state.weights
            w1, w2, w3, w4, w5, w6, bh1, bh2, bo = weights
            X = st.session_state.bp_X
            Y = st.session_state.bp_Y

            sample_idx = st.slider("Pick a sample", 0, len(X) - 1, 0, key="bp_trace_idx")
            x1, x2 = X[sample_idx]
            target = Y[sample_idx]
            st.write(f"**Input:** x1 = {x1}, x2 = {x2} | **Target:** {target}")

            import math
            st.divider()
            st.markdown("**1. Forward Pass — Hidden Layer:**")
            zh1 = x1 * w1 + x2 * w3 + bh1
            zh2 = x1 * w2 + x2 * w4 + bh2
            h1 = 1 / (1 + math.exp(-zh1))
            h2 = 1 / (1 + math.exp(-zh2))
            st.code(f"zh1 = {x1}*{w1:.4f} + {x2}*{w3:.4f} + {bh1:.4f} = {zh1:.4f}\nzh2 = {x1}*{w2:.4f} + {x2}*{w4:.4f} + {bh2:.4f} = {zh2:.4f}\nh1 = sigmoid({zh1:.4f}) = {h1:.4f}\nh2 = sigmoid({zh2:.4f}) = {h2:.4f}")

            st.markdown("**2. Forward Pass — Output:**")
            zo = h1 * w5 + h2 * w6 + bo
            o = 1 / (1 + math.exp(-zo))
            st.code(f"zo = {h1:.4f}*{w5:.4f} + {h2:.4f}*{w6:.4f} + {bo:.4f} = {zo:.4f}\no  = sigmoid({zo:.4f}) = {o:.4f}")

            st.markdown("**3. Error:**")
            error = target - o
            st.code(f"error = {target} - {o:.4f} = {error:.4f}")

            st.markdown("**4. Backpropagation — Compute Deltas:**")
            delta_o = error * o * (1 - o)
            delta_h1 = delta_o * w5 * h1 * (1 - h1)
            delta_h2 = delta_o * w6 * h2 * (1 - h2)
            st.code(f"delta_o  = {error:.4f} * {o:.4f} * {1-o:.4f} = {delta_o:.6f}\ndelta_h1 = {delta_o:.6f} * {w5:.4f} * {h1:.4f} * {1-h1:.4f} = {delta_h1:.6f}\ndelta_h2 = {delta_o:.6f} * {w6:.4f} * {h2:.4f} * {1-h2:.4f} = {delta_h2:.6f}")

            st.markdown("**5. Weight Updates:**")
            lr = b_lr
            st.code(f"w5_new = {w5:.4f} + {lr} * {delta_o:.6f} * {h1:.4f} = {w5 + lr*delta_o*h1:.6f}\nw6_new = {w6:.4f} + {lr} * {delta_o:.6f} * {h2:.4f} = {w6 + lr*delta_o*h2:.6f}\nw1_new = {w1:.4f} + {lr} * {delta_h1:.6f} * {x1} = {w1 + lr*delta_h1*x1:.6f}")
        else:
            st.warning("Train the model first (go to Upload & Train tab).")

    with tab3:
        st.subheader("Visualizations")
        if st.session_state.get("model_type") == "backprop" and st.session_state.get("trained"):
            viz_type = st.selectbox("Select Visualization", [
                "Decision Boundary", "Confidence Heatmap", "Weight Heatmap",
                "Confusion Matrix", "Activation Distribution", "Loss Curve"
            ], key="bp_viz")

            weights = st.session_state.weights
            X_data = st.session_state.get("bp_X")
            Y_data = st.session_state.get("bp_Y")

            if viz_type == "Decision Boundary" and X_data:
                fig = plot_decision_boundary(X_data, Y_data, lambda x1, x2: predict(x1, x2, weights), "Backprop Decision Boundary", theme=_t)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Shows how backpropagation shaped the decision boundary over training.")

            elif viz_type == "Confidence Heatmap" and X_data:
                fig = plot_confidence_heatmap(X_data, Y_data, lambda x1, x2: predict(x1, x2, weights), "Backprop Confidence Map", theme=_t)
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Weight Heatmap":
                fig = plot_weight_heatmap_mlp(weights, "Trained Weight Heatmap", theme=_t)
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Confusion Matrix" and X_data:
                y_pred = [predict(x[0], x[1], weights)[1] for x in X_data]
                fig, metrics = plot_confusion_matrix(Y_data, y_pred, theme=_t)
                st.plotly_chart(fig, use_container_width=True)
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
                mcol2.metric("Precision", f"{metrics['precision']*100:.1f}%")
                mcol3.metric("Recall", f"{metrics['recall']*100:.1f}%")
                mcol4.metric("F1 Score", f"{metrics['f1']*100:.1f}%")

            elif viz_type == "Activation Distribution" and X_data:
                fig = plot_activation_distribution(X_data, weights, theme=_t)
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Loss Curve" and st.session_state.get("loss_history"):
                fig = plot_loss_curve(st.session_state.loss_history, "Backprop Training Loss", theme=_t)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Train with backpropagation first to see visualizations.")

    with tab4:
        st.subheader("Backpropagation Algorithm")
        st.caption("How neural networks learn by propagating errors backward through layers")
        components.html(backprop_diagram(), height=520)

        st.divider()
        st.subheader("What is Backpropagation?")
        st.markdown("""
**Backpropagation** is the training algorithm used to update weights in a neural network. It works by:
1. Running a **forward pass** to get a prediction
2. Computing the **error** (how wrong the prediction is)
3. Propagating the error **backward** through each layer
4. **Updating weights** using the chain rule of calculus

It is NOT a network architecture — it is the **learning algorithm** that can train any feedforward network (MLP, etc.).
""")

        st.divider()
        st.subheader("Step 1: Compute Error")
        st.latex(r"\text{error} = t - o")
        st.latex(r"\text{total\_error} = \sum_{i=1}^{n} (t_i - o_i)^2")
        st.code("""error = target - output
total_error += error ** 2""", language="python")

        st.divider()
        st.subheader("Step 2: Output Layer Gradient (Delta)")
        st.markdown("How much the output neuron contributed to the error:")
        st.latex(r"\delta_o = \text{error} \times o \times (1 - o)")
        st.markdown("This uses the **derivative of sigmoid**: $\\sigma'(x) = \\sigma(x)(1-\\sigma(x))$")
        st.code("delta_o = error * o * (1 - o)", language="python")

        st.divider()
        st.subheader("Step 3: Hidden Layer Gradients (Chain Rule)")
        st.markdown("Error is **propagated backward** through weights to hidden neurons:")
        st.latex(r"\delta_{h1} = \delta_o \times w_5 \times h_1 \times (1 - h_1)")
        st.latex(r"\delta_{h2} = \delta_o \times w_6 \times h_2 \times (1 - h_2)")
        st.markdown("This is the **chain rule** in action — each layer's error depends on the next layer's error.")
        st.code("""delta_h1 = delta_o * w5 * h1 * (1 - h1)
delta_h2 = delta_o * w6 * h2 * (1 - h2)""", language="python")

        st.divider()
        st.subheader("Step 4: Update All Weights")
        st.latex(r"w_{new} = w_{old} + \eta \times \delta \times \text{input}")
        st.markdown("Where $\\eta$ is the learning rate.")
        st.code("""# Output layer weights
w5 = w5 + lr * delta_o * h1
w6 = w6 + lr * delta_o * h2
bo = bo + lr * delta_o

# Hidden layer weights (error flows backward)
w1 = w1 + lr * delta_h1 * x1
w3 = w3 + lr * delta_h1 * x2
w2 = w2 + lr * delta_h2 * x1
w4 = w4 + lr * delta_h2 * x2""", language="python")

        st.divider()
        st.subheader("Why Backpropagation Works")
        st.markdown("""
| Concept | Role |
|---|---|
| **Chain Rule** | Breaks complex derivatives into layer-by-layer products |
| **Gradient Descent** | Uses gradients to move weights toward lower error |
| **Error Signal** | Flows **backward** from output → hidden → input layers |
| **Learning Rate (η)** | Controls step size — too high = overshoot, too low = slow |
""")

# ============================================
# PERCEPTRON
# ============================================
elif model_type == "Perceptron":
    st.header("Perceptron (Single Layer)")
    st.caption("Simple binary classifier with no hidden layers")

    st.sidebar.header("Perceptron Parameters")
    p_lr = st.sidebar.number_input("Learning Rate", min_value=0.00001, max_value=1.0, value=P_DEFAULT_LR, step=0.01, format="%.5f", key="p_lr")
    p_epochs = st.sidebar.number_input("Epochs", min_value=10, max_value=5000, value=P_DEFAULT_EPOCHS, step=10, key="p_epochs")

    tab1, tab2, tab3, tab4 = st.tabs(["Upload & Train", "Predict", "Visualizations", "Architecture"])

    with tab1:
        st.subheader("Step 1: Upload CSV File")
        st.info("CSV format: First 2 columns = features, Last column = label (0 or 1)")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="perceptron_upload")
        use_sample = st.checkbox("Use sample data instead", key="perceptron_sample")

        if use_sample:
            df = pd.read_csv("sample_data.csv")
            st.write("Sample Data (Student Performance):")
            st.dataframe(df, use_container_width=True)
            X = df.iloc[:, :2].values.tolist()
            Y = df.iloc[:, -1].values.tolist()
            st.session_state.col_names = [df.columns[0], df.columns[1]]
        elif uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:")
            st.dataframe(df, use_container_width=True)
            X = df.iloc[:, :2].values.tolist()
            Y = df.iloc[:, -1].values.tolist()
            st.session_state.col_names = [df.columns[0], df.columns[1]]
        else:
            X = None
            Y = None

        st.subheader("Step 2: Train Perceptron")

        if X is not None and Y is not None:
            if st.button("Train Perceptron", type="primary"):
                with st.spinner("Training..."):
                    weights, loss_history = train_perceptron(X, Y, lr=p_lr, epochs=p_epochs)

                st.session_state.trained = True
                st.session_state.weights = weights
                st.session_state.loss_history = loss_history
                st.session_state.model_type = "perceptron"
                st.session_state.perceptron_X = X
                st.session_state.perceptron_Y = Y

                st.success("Training Complete!")

                w1, w2, b = weights
                st.subheader("Trained Weights")
                st.write(f"w1: {w1:.6f}")
                st.write(f"w2: {w2:.6f}")
                st.write(f"bias: {b:.6f}")

                st.subheader("Training Loss Over Epochs")
                loss_df = pd.DataFrame({'Epoch': range(1, len(loss_history) + 1), 'Errors': loss_history})
                st.line_chart(loss_df.set_index('Epoch'))
        else:
            st.warning("Please upload a CSV file or use sample data.")

    with tab2:
        st.subheader("Predict Output")

        if st.session_state.trained and st.session_state.model_type == "perceptron":
            st.success("Model is trained and ready!")
            col_names = st.session_state.get('col_names', ['Feature 1', 'Feature 2'])

            pcol1, pcol2 = st.columns(2)
            with pcol1:
                pred_x1 = st.number_input(col_names[0], value=5.0, key="p_x1")
            with pcol2:
                pred_x2 = st.number_input(col_names[1], value=75.0, key="p_x2")

            if st.button("Predict", type="primary", key="p_predict"):
                z, pred = predict_perceptron(pred_x1, pred_x2, st.session_state.weights)

                st.subheader("Prediction Result")
                st.metric("Raw Output (z)", f"{z:.6f}")
                st.metric("Predicted Class", "Pass" if pred == 1 else "Fail")
        else:
            st.warning("Please train the perceptron first.")

    with tab3:
        st.subheader("Visualizations")
        if st.session_state.get("model_type") == "perceptron" and st.session_state.get("trained"):
            viz_type = st.selectbox("Select Visualization", [
                "Decision Boundary", "Confusion Matrix", "Loss Curve"
            ], key="p_viz")

            weights = st.session_state.weights
            X_data = st.session_state.get("perceptron_X")
            Y_data = st.session_state.get("perceptron_Y")

            if viz_type == "Decision Boundary" and X_data:
                fig = plot_perceptron_boundary(X_data, Y_data, weights, "Perceptron Decision Boundary", theme=_t)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("The perceptron finds a single straight line to separate the two classes.")

            elif viz_type == "Confusion Matrix" and X_data:
                y_pred = [predict_perceptron(x[0], x[1], weights)[1] for x in X_data]
                fig, metrics = plot_confusion_matrix(Y_data, y_pred, "Perceptron Confusion Matrix", theme=_t)
                st.plotly_chart(fig, use_container_width=True)
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
                mcol2.metric("Precision", f"{metrics['precision']*100:.1f}%")
                mcol3.metric("Recall", f"{metrics['recall']*100:.1f}%")
                mcol4.metric("F1 Score", f"{metrics['f1']*100:.1f}%")

            elif viz_type == "Loss Curve" and st.session_state.get("loss_history"):
                fig = plot_loss_curve(st.session_state.loss_history, "Perceptron Training Errors", ylabel="Misclassifications", theme=_t)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Train the perceptron first to see visualizations.")

    with tab4:
        st.subheader("Network Diagram")
        st.caption("Drag nodes to move, scroll to zoom, drag background to pan")
        components.html(perceptron_diagram(), height=520)

        st.divider()
        st.subheader("Step 1: Input Preparation")
        st.markdown("Add bias column (always `1`) to inputs. Convert binary targets to **bipolar**:")
        st.latex(r"\text{target} = \begin{cases} +1 & \text{if } y = 1 \\ -1 & \text{if } y = 0 \end{cases}")
        st.code("""inputs = np.array([[x[0], x[1], 1] for x in X])
targets = np.array([1 if y == 1 else -1 for y in Y])
weights = np.array([0.0, 0.0, 0.0])""", language="python")

        st.divider()
        st.subheader("Step 2: Compute Weighted Sum")
        st.latex(r"Y_{in} = x_1 \cdot w_1 + x_2 \cdot w_2 + 1 \cdot w_b = \mathbf{x} \cdot \mathbf{w}")
        st.code("""Y_in = np.dot(x, weights)""", language="python")

        st.divider()
        st.subheader("Step 3: Step Activation Function")
        st.latex(r"Y_{out} = \begin{cases} 1 & \text{if } Y_{in} > 0 \\ 0 & \text{otherwise} \end{cases}")
        st.code("""if Y_in > 0:
    Y_out = 1
else:
    Y_out = 0""", language="python")

        st.divider()
        st.subheader("Step 4: Check Match (Bipolar Logic)")
        st.markdown("Compare prediction with bipolar target:")
        st.latex(r"\text{match} = (t = +1 \text{ and } Y_{out} = 1) \text{ or } (t = -1 \text{ and } Y_{out} = 0)")
        st.code("""is_match = (t == 1 and Y_out == 1) or (t == -1 and Y_out == 0)""", language="python")

        st.divider()
        st.subheader("Step 5: Update Weights (Only on Mismatch)")
        st.latex(r"\mathbf{w}_{new} = \mathbf{w}_{old} + \eta \cdot t \cdot \mathbf{x}")
        st.markdown("Where $\\eta$ is the learning rate and $t$ is the bipolar target.")
        st.code("""if not is_match:
    weights = weights + learning_rate * (t * x)""", language="python")

        st.divider()
        st.subheader("Final Decision Boundary")
        st.latex(r"w_1 \cdot x_1 + w_2 \cdot x_2 + w_b = 0")

# ============================================
# RNN (Sentiment Analysis)
# ============================================
elif model_type == "RNN (Sentiment Analysis)":
    st.header("RNN — Sentiment Analysis")
    st.caption("Vanilla RNN with word embeddings for binary text classification")

    st.sidebar.header("RNN Parameters")
    r_hidden = st.sidebar.number_input("Hidden Size", min_value=8, max_value=256, value=R_DEFAULT_HIDDEN, step=8, key="r_hidden")
    r_lr = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=R_DEFAULT_LR, step=0.001, format="%.4f", key="r_lr")
    r_epochs = st.sidebar.number_input("Epochs", min_value=5, max_value=500, value=R_DEFAULT_EPOCHS, step=5, key="r_epochs")

    tab1, tab2, tab3, tab4 = st.tabs(["Upload & Train", "Predict", "Visualizations", "Architecture"])

    # ── Tab 1: Upload & Train ────────────────────────────────
    with tab1:
        st.subheader("Step 1: Upload Sentiment CSV")
        st.info("CSV format: A **text** column and a **sentiment** column (0 = Negative, 1 = Positive)")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="rnn_upload")
        use_sample = st.checkbox("Use sample sentiment data instead", key="rnn_sample")

        if use_sample:
            df = pd.read_csv("sample_sentiment.csv")
            st.write("Sample Sentiment Data:")
            st.dataframe(df, use_container_width=True)
            texts = df["text"].tolist()
            labels = df["sentiment"].tolist()
        elif uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:")
            st.dataframe(df, use_container_width=True)
            # Auto-detect text and label columns
            text_col = st.selectbox("Select text column", df.columns, index=0, key="rnn_text_col")
            label_col = st.selectbox("Select label column (0/1)", df.columns, index=min(1, len(df.columns) - 1), key="rnn_label_col")
            texts = df[text_col].astype(str).tolist()
            labels = df[label_col].astype(int).tolist()
        else:
            texts = None
            labels = None

        st.subheader("Step 2: Train RNN")

        if texts is not None and labels is not None:
            pos_count = sum(labels)
            neg_count = len(labels) - pos_count
            st.write(f"**Dataset:** {len(labels)} samples — {pos_count} Positive, {neg_count} Negative")

            if st.button("Train RNN", type="primary", key="rnn_train"):
                with st.spinner("Training RNN (this may take a moment)..."):
                    model, vocab, loss_hist, acc_hist = train_rnn(
                        texts, labels,
                        hidden_size=r_hidden, lr=r_lr, epochs=r_epochs
                    )
                    st.session_state.trained = True
                    st.session_state.rnn_model = model
                    st.session_state.rnn_vocab = vocab
                    st.session_state.loss_history = loss_hist
                    st.session_state.rnn_acc_history = acc_hist
                    st.session_state.model_type = "rnn"
                    st.session_state.rnn_texts = texts
                    st.session_state.rnn_labels = labels

                st.success("Training Complete!")

                st.write(f"**Vocab size:** {len(vocab)} words  |  **Hidden size:** {r_hidden}")

                col_l, col_r = st.columns(2)
                with col_l:
                    st.subheader("Training Loss")
                    loss_df = pd.DataFrame({"Epoch": range(1, len(loss_hist) + 1), "Loss": loss_hist})
                    st.line_chart(loss_df.set_index("Epoch"))
                with col_r:
                    st.subheader("Training Accuracy")
                    acc_df = pd.DataFrame({"Epoch": range(1, len(acc_hist) + 1), "Accuracy": acc_hist})
                    st.line_chart(acc_df.set_index("Epoch"))

                # Show final accuracy
                final_acc = acc_hist[-1] * 100
                st.metric("Final Training Accuracy", f"{final_acc:.1f}%")
        else:
            st.warning("Please upload a CSV file or use sample data.")

    # ── Tab 2: Predict ───────────────────────────────────────
    with tab2:
        st.subheader("Predict Sentiment")

        if st.session_state.get("model_type") == "rnn" and st.session_state.get("trained"):
            st.success("RNN model is trained and ready!")

            input_text = st.text_area("Enter text to analyse:", height=100, key="rnn_input",
                                       placeholder="e.g. This product is amazing and I love it!")

            if st.button("Analyse Sentiment", type="primary", key="rnn_predict"):
                if input_text.strip():
                    model = st.session_state.rnn_model
                    vocab = st.session_state.rnn_vocab
                    score, label = model.predict_text(input_text, vocab)

                    st.subheader("Result")
                    rcol1, rcol2 = st.columns(2)
                    with rcol1:
                        if label == "Positive":
                            st.success(f"**{label}**")
                        else:
                            st.error(f"**{label}**")
                    with rcol2:
                        st.metric("Confidence Score", f"{score:.4f}")

                    st.progress(score)
                    st.caption(f"Score: {score:.4f}  (>= 0.5 → Positive, < 0.5 → Negative)")
                else:
                    st.warning("Please enter some text.")

            st.divider()
            st.subheader("Batch Prediction")
            st.info("Upload a CSV with a text column to predict sentiment for many rows at once.")
            batch_file = st.file_uploader("Upload CSV for batch prediction", type="csv", key="rnn_batch")
            if batch_file is not None:
                batch_df = pd.read_csv(batch_file)
                batch_col = st.selectbox("Select text column", batch_df.columns, key="rnn_batch_col")
                if st.button("Run Batch Prediction", key="rnn_batch_run"):
                    model = st.session_state.rnn_model
                    vocab = st.session_state.rnn_vocab
                    scores = []
                    preds = []
                    for t in batch_df[batch_col].astype(str):
                        s, l = model.predict_text(t, vocab)
                        scores.append(round(s, 4))
                        preds.append(l)
                    batch_df["Score"] = scores
                    batch_df["Prediction"] = preds
                    st.dataframe(batch_df, use_container_width=True)

                    pos = preds.count("Positive")
                    neg = preds.count("Negative")
                    st.write(f"**Summary:** {pos} Positive, {neg} Negative out of {len(preds)} texts")
        else:
            st.warning("Please train the RNN first.")

    # ── Tab 3: Visualizations ────────────────────────────────
    with tab3:
        st.subheader("Visualizations")
        if st.session_state.get("model_type") == "rnn" and st.session_state.get("trained"):
            viz_type = st.selectbox("Select Visualization", [
                "Sentiment Distribution", "Confusion Matrix", "Loss & Accuracy Curves", "Word Cloud Preview"
            ], key="rnn_viz")

            if viz_type == "Sentiment Distribution":
                labels_data = st.session_state.get("rnn_labels", [])
                fig = plot_sentiment_distribution(labels_data, theme=_t)
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Confusion Matrix":
                model = st.session_state.rnn_model
                vocab = st.session_state.rnn_vocab
                texts_data = st.session_state.get("rnn_texts", [])
                labels_data = st.session_state.get("rnn_labels", [])
                y_pred = []
                for t in texts_data:
                    _, lbl = model.predict_text(t, vocab)
                    y_pred.append(1 if lbl == "Positive" else 0)
                fig, metrics = plot_confusion_matrix(labels_data, y_pred, "RNN Confusion Matrix", theme=_t)
                st.plotly_chart(fig, use_container_width=True)
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
                mcol2.metric("Precision", f"{metrics['precision']*100:.1f}%")
                mcol3.metric("Recall", f"{metrics['recall']*100:.1f}%")
                mcol4.metric("F1 Score", f"{metrics['f1']*100:.1f}%")

            elif viz_type == "Loss & Accuracy Curves":
                loss_hist = st.session_state.get("loss_history", [])
                acc_hist = st.session_state.get("rnn_acc_history", [])
                fig = plot_loss_accuracy(loss_hist, acc_hist, theme=_t)
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "Word Cloud Preview":
                texts_data = st.session_state.get("rnn_texts", [])
                labels_data = st.session_state.get("rnn_labels", [])
                from rnn import tokenize
                fig = plot_word_frequency(texts_data, labels_data, tokenize, theme=_t)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Train the RNN first to see visualizations.")

    # ── Tab 4: Architecture ──────────────────────────────────
    with tab4:
        st.subheader("Network Diagram")
        st.caption("Drag nodes to move, scroll to zoom")
        components.html(rnn_diagram(), height=520)

        st.divider()
        st.subheader("Step 1: Word Embedding")
        st.markdown("Each word is mapped to a dense vector via an embedding matrix:")
        st.latex(r"\mathbf{x}_t = \mathbf{W}_{embed}[\text{word}_t]")
        st.code('x_t = W_embed[word_index]  # shape: (1, embed_size)', language="python")

        st.divider()
        st.subheader("Step 2: Recurrent Hidden State")
        st.markdown("At each time step, the hidden state is updated:")
        st.latex(r"\mathbf{h}_t = \tanh(\mathbf{x}_t \cdot \mathbf{W}_{xh} + \mathbf{h}_{t-1} \cdot \mathbf{W}_{hh} + \mathbf{b}_h)")
        st.code("""z_t = x_t @ W_xh + h_{t-1} @ W_hh + b_h
h_t = tanh(z_t)""", language="python")

        st.divider()
        st.subheader("Step 3: Output (Many-to-One)")
        st.markdown("After processing all words, the final hidden state produces the output:")
        st.latex(r"y = \sigma(\mathbf{h}_T \cdot \mathbf{W}_{hy} + \mathbf{b}_y)")
        st.code("""logit = h_T @ W_hy + b_y
y = sigmoid(logit)  # 0..1 probability""", language="python")

        st.divider()
        st.subheader("Step 4: Loss (Binary Cross-Entropy)")
        st.latex(r"\mathcal{L} = -\left[ y_{true} \cdot \log(y) + (1 - y_{true}) \cdot \log(1 - y) \right]")
        st.code('loss = -(y_true * log(y) + (1 - y_true) * log(1 - y))', language="python")

        st.divider()
        st.subheader("Step 5: Backpropagation Through Time (BPTT)")
        st.markdown("Gradients flow backward through each time step:")
        st.latex(r"\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{xh}} = \sum_{t=1}^{T} \mathbf{x}_t^T \cdot \delta_t")
        st.latex(r"\delta_t = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} \odot (1 - \tanh^2(\mathbf{z}_t))")
        st.code("""for t in reversed(range(T)):
    dtanh = dh * (1 - tanh(z_t)^2)
    dW_xh += x_t.T @ dtanh
    dW_hh += h_{t-1}.T @ dtanh
    dh = dtanh @ W_hh.T  # propagate to previous step""", language="python")

# ============================================
# MSE LOSS (Linear Regression)
# ============================================
elif model_type == "MSE Loss (Linear Regression)":
    st.header("MSE Loss — Linear Regression")
    st.caption("Gradient descent with Mean Squared Error loss")

    st.sidebar.header("MSE Parameters")
    mse_mode = st.sidebar.radio("Input Mode", ["Single Variable (1 feature)", "Two Variables (2 features)"], key="mse_mode")
    m_lr = st.sidebar.number_input("Learning Rate", min_value=0.00001, max_value=1.0, value=M_DEFAULT_LR, step=0.001, format="%.5f", key="m_lr")
    m_epochs = st.sidebar.number_input("Epochs", min_value=10, max_value=5000, value=M_DEFAULT_EPOCHS, step=10, key="m_epochs")

    tab1, tab2, tab3, tab4 = st.tabs(["Upload & Train", "Predict", "Visualizations", "Architecture"])

    # ── Tab 1: Upload & Train ────────────────────────────────
    with tab1:
        st.subheader("Step 1: Upload CSV File")

        if mse_mode == "Single Variable (1 feature)":
            st.info("CSV format: First column = feature (X), Last column = target (y)")
        else:
            st.info("CSV format: First 2 columns = features (X1, X2), Last column = target (y)")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="mse_upload")
        use_sample = st.checkbox("Use sample data instead", key="mse_sample")

        if use_sample:
            if mse_mode == "Single Variable (1 feature)":
                # Real-world: House size (1000 sqft) → Price (in $1000s)
                X_single = [0.8, 1.0, 1.2, 1.4, 1.5, 1.7, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.2, 3.5, 3.8, 4.0, 4.5, 5.0]
                y_vals =   [150, 180, 200, 230, 245, 270, 290, 320, 350, 400, 440, 470, 510, 550, 590, 620, 700, 780]
                df = pd.DataFrame({"House_Size_1000sqft": X_single, "Price_1000USD": y_vals})
                st.write("Sample Data — House Price vs Size:")
                st.dataframe(df, use_container_width=True)
                st.session_state.mse_col_names = ["House_Size_1000sqft"]
            else:
                # Real-world: House size + bedrooms → Price
                X1_vals = [0.8, 1.0, 1.2, 1.4, 1.5, 1.7, 2.0, 2.2, 2.5, 2.8, 3.0, 3.2, 3.5, 3.8, 4.0, 4.5, 5.0, 5.5]
                X2_vals = [1,   1,   2,   2,   2,   2,   3,   3,   3,   3,   4,   4,   4,   4,   5,   5,   5,   6]
                y_vals  = [150, 180, 210, 240, 250, 280, 340, 370, 410, 450, 490, 520, 560, 600, 640, 720, 800, 880]
                df = pd.DataFrame({"House_Size_1000sqft": X1_vals, "Bedrooms": X2_vals, "Price_1000USD": y_vals})
                st.write("Sample Data — House Price vs Size & Bedrooms:")
                st.dataframe(df, use_container_width=True)
                st.session_state.mse_col_names = ["House_Size_1000sqft", "Bedrooms"]
        elif uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data:")
            st.dataframe(df)
            y_vals = df.iloc[:, -1].values.tolist()
            if mse_mode == "Single Variable (1 feature)":
                X_single = df.iloc[:, 0].values.tolist()
                st.session_state.mse_col_names = [df.columns[0]]
            else:
                X1_vals = df.iloc[:, 0].values.tolist()
                X2_vals = df.iloc[:, 1].values.tolist()
                st.session_state.mse_col_names = [df.columns[0], df.columns[1]]
        else:
            df = None
            y_vals = None

        st.subheader("Step 2: Train Model")

        if df is not None and y_vals is not None:
            if st.button("Train", type="primary", key="mse_train"):
                with st.spinner("Training..."):
                    if mse_mode == "Single Variable (1 feature)":
                        w, b, init_w, init_b, loss_hist = train_mse_single(
                            X_single, y_vals, learning_rate=m_lr, epochs=m_epochs
                        )
                        st.session_state.trained = True
                        st.session_state.mse_weights = (w, b)
                        st.session_state.loss_history = loss_hist
                        st.session_state.model_type = "mse_single"
                        st.session_state.mse_X_single = X_single
                        st.session_state.mse_y = y_vals

                        st.success("Training Complete!")

                        st.subheader("Initial Random Weights")
                        st.write(f"w: {init_w:.6f},  b: {init_b:.6f}")

                        st.subheader("Trained Weights")
                        st.write(f"w: {w:.6f},  b: {b:.6f}")

                    else:
                        w1, w2, b, iw1, iw2, ib, loss_hist = train_mse_dual(
                            X1_vals, X2_vals, y_vals, learning_rate=m_lr, epochs=m_epochs
                        )
                        st.session_state.trained = True
                        st.session_state.mse_weights = (w1, w2, b)
                        st.session_state.loss_history = loss_hist
                        st.session_state.model_type = "mse_dual"
                        st.session_state.mse_X1 = X1_vals
                        st.session_state.mse_X2 = X2_vals
                        st.session_state.mse_y = y_vals

                        st.success("Training Complete!")

                        st.subheader("Initial Random Weights")
                        st.write(f"w1: {iw1:.6f},  w2: {iw2:.6f},  b: {ib:.6f}")

                        st.subheader("Trained Weights")
                        wcol1, wcol2, wcol3 = st.columns(3)
                        with wcol1:
                            st.write(f"w1: {w1:.6f}")
                        with wcol2:
                            st.write(f"w2: {w2:.6f}")
                        with wcol3:
                            st.write(f"b: {b:.6f}")

                    st.subheader("Training Loss (MSE) Over Epochs")
                    loss_df = pd.DataFrame({"Epoch": range(1, len(loss_hist) + 1), "MSE Loss": loss_hist})
                    st.line_chart(loss_df.set_index("Epoch"))

                    st.metric("Final MSE Loss", f"{loss_hist[-1]:.6f}")
        else:
            st.warning("Please upload a CSV file or use sample data.")

    # ── Tab 2: Predict ───────────────────────────────────────
    with tab2:
        st.subheader("Predict Output")

        if st.session_state.get("trained") and st.session_state.get("model_type", "").startswith("mse"):
            st.success("Model is trained and ready!")
            col_names = st.session_state.get("mse_col_names", ["Feature 1", "Feature 2"])

            if st.session_state.model_type == "mse_single":
                pred_x = st.number_input(col_names[0], value=6.0, key="mse_pred_x")
                if st.button("Predict", type="primary", key="mse_predict"):
                    w, b = st.session_state.mse_weights
                    result = predict_single(pred_x, w, b)
                    st.subheader("Prediction Result")
                    st.metric("Predicted Value (ŷ)", f"{result:.4f}")
                    st.caption(f"ŷ = {w:.4f} × {pred_x} + {b:.4f} = {result:.4f}")
            else:
                pcol1, pcol2 = st.columns(2)
                with pcol1:
                    pred_x1 = st.number_input(col_names[0], value=6.0, key="mse_pred_x1")
                with pcol2:
                    pred_x2 = st.number_input(col_names[1], value=80.0, key="mse_pred_x2")
                if st.button("Predict", type="primary", key="mse_predict"):
                    w1, w2, b = st.session_state.mse_weights
                    result = predict_dual(pred_x1, pred_x2, w1, w2, b)
                    st.subheader("Prediction Result")
                    st.metric("Predicted Value (ŷ)", f"{result:.4f}")
                    st.caption(f"ŷ = {w1:.4f}×{pred_x1} + {w2:.4f}×{pred_x2} + {b:.4f} = {result:.4f}")
        else:
            st.warning("Please train the model first.")

    # ── Tab 3: Visualizations ────────────────────────────────
    with tab3:
        st.subheader("Visualizations")
        if st.session_state.get("trained") and st.session_state.get("model_type", "").startswith("mse"):
            viz_type = st.selectbox("Select Visualization", [
                "Regression Fit", "Residual Plot", "Loss Curve"
            ], key="mse_viz")

            if viz_type == "Regression Fit":
                if st.session_state.model_type == "mse_single":
                    w, b = st.session_state.mse_weights
                    X_data = st.session_state.get("mse_X_single", [])
                    y_data = st.session_state.get("mse_y", [])
                    fig = plot_regression_line(X_data, y_data, w, b, "Linear Regression Fit (Single Variable)", theme=_t)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Blue dots = actual data. Red line = fitted model. Gray dashed = residuals (errors).")
                else:
                    w1, w2, b = st.session_state.mse_weights
                    X1_data = st.session_state.get("mse_X1", [])
                    X2_data = st.session_state.get("mse_X2", [])
                    y_data = st.session_state.get("mse_y", [])
                    fig = plot_regression_3d(X1_data, X2_data, y_data, w1, w2, b, "3D Regression Surface", theme=_t)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Blue dots = actual data. Surface = fitted plane. Drag to rotate in 3D.")

            elif viz_type == "Residual Plot":
                y_data = st.session_state.get("mse_y", [])
                if st.session_state.model_type == "mse_single":
                    w, b = st.session_state.mse_weights
                    X_data = st.session_state.get("mse_X_single", [])
                    y_pred = [predict_single(x, w, b) for x in X_data]
                else:
                    w1, w2, b = st.session_state.mse_weights
                    X1_data = st.session_state.get("mse_X1", [])
                    X2_data = st.session_state.get("mse_X2", [])
                    y_pred = [predict_dual(X1_data[i], X2_data[i], w1, w2, b) for i in range(len(X1_data))]
                fig = plot_residual(y_data, y_pred, theme=_t)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Points near zero = good predictions. Patterns suggest model is missing something.")

            elif viz_type == "Loss Curve" and st.session_state.get("loss_history"):
                fig = plot_loss_curve(st.session_state.loss_history, "MSE Training Loss", ylabel="MSE", theme=_t)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Train the model first to see visualizations.")

    # ── Tab 4: Architecture ──────────────────────────────────
    with tab4:
        st.subheader("Network Diagram")
        st.caption("Drag nodes to move, scroll to zoom")
        components.html(mse_diagram(), height=520)

        st.divider()
        st.subheader("Step 1: Prediction (Forward Pass)")
        st.markdown("**Single variable:**")
        st.latex(r"\hat{y} = w \cdot x + b")
        st.markdown("**Two variables:**")
        st.latex(r"\hat{y} = w_1 \cdot x_1 + w_2 \cdot x_2 + b")
        st.code("""y_pred = w1 * x1 + w2 * x2 + b""", language="python")

        st.divider()
        st.subheader("Step 2: MSE Loss")
        st.latex(r"\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2")
        st.code("""loss = sum((y_pred - y_true)^2) / n""", language="python")

        st.divider()
        st.subheader("Step 3: Compute Gradients")
        st.latex(r"\frac{\partial \text{MSE}}{\partial w} = \frac{2}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i) \cdot x_i")
        st.latex(r"\frac{\partial \text{MSE}}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)")
        st.code("""for i in range(n):
    error = y_pred - y[i]
    dw += error * x[i]
    db += error
dw = (2 / n) * dw
db = (2 / n) * db""", language="python")

        st.divider()
        st.subheader("Step 4: Update Weights (Gradient Descent)")
        st.latex(r"w = w - \eta \cdot \frac{\partial \text{MSE}}{\partial w}")
        st.latex(r"b = b - \eta \cdot \frac{\partial \text{MSE}}{\partial b}")
        st.code("""w = w - learning_rate * dw
b = b - learning_rate * db""", language="python")
