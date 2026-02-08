"""
Gaussian Process Explorer 

Run with: streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, Any, Tuple

from kernels import (
    Kernel, KERNEL_REGISTRY, get_kernel_names, create_kernel,
    SumKernel, ProductKernel
)
from gpr import GPR


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="GPExplorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stSidebar"] {min-width: 380px; max-width: 450px;}
    .block-container {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Defaults (from notebook)
# =============================================================================

DEFAULT_X = np.array([0.0, 0.3, 1.0, 3.1, 4.7])
DEFAULT_Y = np.array([1.0, 0.0, 1.4, 0.0, -0.9])


# =============================================================================
# Helpers
# =============================================================================

def render_kernel_params(kernel_name: str, prefix: str = "") -> Dict[str, Any]:
    """Render sliders for kernel parameters."""
    kernel_class = KERNEL_REGISTRY[kernel_name]
    param_info = kernel_class.param_info()
    params = {}
    
    cols = st.columns(len(param_info)) if len(param_info) <= 3 else [st.container()]
    
    for i, (param_name, info) in enumerate(param_info.items()):
        col = cols[i] if len(param_info) <= 3 else cols[0]
        with col:
            if isinstance(info['default'], int):
                params[param_name] = st.slider(
                    info['description'], int(info['min']), int(info['max']),
                    int(info['default']), int(info['step']), key=f"{prefix}{param_name}"
                )
            else:
                params[param_name] = st.slider(
                    info['description'], float(info['min']), float(info['max']),
                    float(info['default']), float(info['step']), key=f"{prefix}{param_name}"
                )
    return params


def build_kernel(state) -> Tuple[Kernel, str, str]:
    """Build kernel from session state, return (kernel, name, latex)."""
    if state.get('composite_mode', False):
        k1 = create_kernel(state['kernel1_name'], **state['kernel1_params'])
        k2 = create_kernel(state['kernel2_name'], **state['kernel2_params'])
        op = state['composition_operator']
        
        if op == '+':
            kernel = SumKernel(k1, k2)
            name = f"{state['kernel1_name']} + {state['kernel2_name']}"
        else:
            kernel = ProductKernel(k1, k2)
            name = f"{state['kernel1_name']} × {state['kernel2_name']}"
        
        k1_latex = KERNEL_REGISTRY[state['kernel1_name']].latex_formula()
        k2_latex = KERNEL_REGISTRY[state['kernel2_name']].latex_formula()
        latex = f"k_1 {op} k_2 \\quad\\text{{where}}\\quad k_1: {k1_latex}, \\quad k_2: {k2_latex}"
        return kernel, name, latex
    else:
        k_name = state['kernel_name']
        kernel = create_kernel(k_name, **state['kernel_params'])
        return kernel, k_name, KERNEL_REGISTRY[k_name].latex_formula()


def create_gpr_plot(model: GPR, x_pred: np.ndarray, x_train: np.ndarray, 
                    y_train: np.ndarray, title: str) -> go.Figure:
    """Main GPR visualization."""
    mean = model.predict(x_pred)
    std = np.sqrt(model._memory['variance'])
    
    fig = go.Figure()
    
    # Uncertainty bands
    for i, (sigma, color) in enumerate([(3, 'rgba(100,149,237,0.12)'), 
                                         (2, 'rgba(100,149,237,0.22)'),
                                         (1, 'rgba(100,149,237,0.35)')]):
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_pred, x_pred[::-1]]),
            y=np.concatenate([mean + sigma*std, (mean - sigma*std)[::-1]]),
            fill='toself', fillcolor=color, line=dict(color='rgba(0,0,0,0)'),
            hoverinfo='skip', showlegend=True, name=f'±{sigma}σ'
        ))
    
    # Mean
    fig.add_trace(go.Scatter(x=x_pred, y=mean, mode='lines',
        line=dict(color='#1E3A5F', width=2.5), name='Mean'))
    
    # Observations
    fig.add_trace(go.Scatter(x=x_train, y=y_train, mode='markers',
        marker=dict(color='#DC143C', size=11, line=dict(width=1.5, color='white')),
        name='Data'))
    
    fig.update_layout(
        title=dict(text=title, x=0.5), height=480, plot_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        margin=dict(l=50, r=30, t=80, b=50),
        xaxis_title='x', yaxis_title='y'
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor='#333', gridcolor='#eee')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='#333', gridcolor='#eee')
    return fig


def create_sample_plot(model: GPR, x_pred: np.ndarray, x_train: np.ndarray,
                       y_train: np.ndarray, n_samples: int) -> go.Figure:
    """Random function samples from posterior."""
    samples = model.sample(x_pred, n_samples=n_samples)
    fig = go.Figure()
    
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', 
              '#19D3F3', '#FF6692', '#B6E880']
    for i, sample in enumerate(samples):
        fig.add_trace(go.Scatter(x=x_pred, y=sample, mode='lines',
            line=dict(color=colors[i % len(colors)], width=1.5),
            name=f'Sample {i+1}', opacity=0.75))
    
    fig.add_trace(go.Scatter(x=x_train, y=y_train, mode='markers',
        marker=dict(color='#DC143C', size=9, line=dict(width=1, color='white')),
        name='Data'))
    
    fig.update_layout(
        title=dict(text='Posterior Samples', x=0.5), height=350, plot_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        margin=dict(l=50, r=30, t=70, b=40), xaxis_title='x', yaxis_title='y'
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor='#333', gridcolor='#eee')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='#333', gridcolor='#eee')
    return fig


def create_kernel_heatmap(model: GPR) -> go.Figure:
    """Kernel matrix heatmap."""
    K = model.train_kernel_matrix
    n = K.shape[0]
    fig = go.Figure(data=go.Heatmap(z=K, colorscale='Viridis', colorbar=dict(title="k(xi,xj)")))
    fig.update_layout(
        title=dict(text='Covariance Matrix K(X,X)', x=0.5), height=320,
        xaxis_title='Point index i', yaxis_title='Point index j',
        margin=dict(l=40, r=30, t=50, b=40), yaxis=dict(autorange='reversed')
    )
    return fig


# =============================================================================
# Sidebar Controls
# =============================================================================

with st.sidebar:
    st.header("Kernel")
    
    composite_mode = st.toggle("Combine two kernels", key='composite_mode')
    
    if composite_mode:
        col1, col2, col3 = st.columns([5, 2, 5])
        with col1:
            kernel1_name = st.selectbox("Primary", get_kernel_names(), index=0, key='kernel1_name')
        with col2:
            st.write("")
            st.write("")
            composition_operator = st.radio("Op", ['+', '×'], horizontal=True, 
                                            key='composition_operator', label_visibility='collapsed')
        with col3:
            kernel2_name = st.selectbox("Secondary", get_kernel_names(), index=4, key='kernel2_name')
        
        st.markdown("##### Primary kernel parameters")
        kernel1_params = render_kernel_params(kernel1_name, prefix='k1_')
        st.session_state['kernel1_params'] = kernel1_params
        
        st.markdown("##### Secondary kernel parameters")
        kernel2_params = render_kernel_params(kernel2_name, prefix='k2_')
        st.session_state['kernel2_params'] = kernel2_params
    else:
        kernel_name = st.selectbox("Type", get_kernel_names(), index=0, key='kernel_name')
        kernel_params = render_kernel_params(kernel_name, prefix='k_')
        st.session_state['kernel_params'] = kernel_params
    
    st.markdown("---")
    
    # Noise
    st.header("Noise")
    noise_sigma = st.slider("Observation noise σ", 0.0, 1.0, 0.0, 0.01, key='noise_sigma')
    
    st.markdown("---")
    
    # Data
    st.header("Data")
    
    data_source = st.radio("Source", ["Default", "Custom", "Upload CSV"], 
                           horizontal=True, key='data_source')
    
    if data_source == "Default":
        x_train = DEFAULT_X.copy()
        y_train = DEFAULT_Y.copy()
        st.caption(f"{len(x_train)} points: x ∈ [{x_train.min():.1f}, {x_train.max():.1f}]")
        
    elif data_source == "Custom":
        n_points = st.number_input("Points", 2, 20, len(DEFAULT_X), key='n_points')
        
        if 'custom_df' not in st.session_state or len(st.session_state.custom_df) != n_points:
            if n_points <= len(DEFAULT_X):
                st.session_state.custom_df = pd.DataFrame({'x': DEFAULT_X[:n_points], 'y': DEFAULT_Y[:n_points]})
            else:
                extra = n_points - len(DEFAULT_X)
                # Random points within reasonable range
                np.random.seed(None)  # Truly random
                rand_x = np.random.uniform(x_train.min() - 1 if 'x_train' in dir() else -1, 
                                           x_train.max() + 2 if 'x_train' in dir() else 6, extra)
                rand_y = np.random.uniform(-2, 2, extra)
                st.session_state.custom_df = pd.DataFrame({
                    'x': np.concatenate([DEFAULT_X, rand_x]),
                    'y': np.concatenate([DEFAULT_Y, rand_y])
                })
        
        edited_df = st.data_editor(st.session_state.custom_df, num_rows='fixed', key='data_editor')
        x_train = edited_df['x'].values.astype(float)
        y_train = edited_df['y'].values.astype(float)
        
    else:  # Upload CSV
        uploaded = st.file_uploader("CSV with 'x' and 'y' columns", type=['csv'], key='csv_upload')
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                if 'x' in df.columns and 'y' in df.columns:
                    x_train = df['x'].values.astype(float)
                    y_train = df['y'].values.astype(float)
                    st.success(f"Loaded {len(x_train)} points")
                else:
                    st.error("CSV must have 'x' and 'y' columns")
                    x_train, y_train = DEFAULT_X.copy(), DEFAULT_Y.copy()
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                x_train, y_train = DEFAULT_X.copy(), DEFAULT_Y.copy()
        else:
            st.info("Upload a CSV file")
            x_train, y_train = DEFAULT_X.copy(), DEFAULT_Y.copy()
    
    st.markdown("---")
    
    # Prediction range
    st.header("Prediction Range")
    col1, col2 = st.columns(2)
    with col1:
        x_min = st.number_input("Min", value=float(x_train.min() - 1), key='x_min')
    with col2:
        x_max = st.number_input("Max", value=float(x_train.max() + 1), key='x_max')


# =============================================================================
# Main Content (Single Scrollable Page)
# =============================================================================

# Build kernel
kernel, kernel_name_display, kernel_latex = build_kernel(st.session_state)

# Prediction grid
x_pred = np.arange(x_min, x_max, 0.05)

# Fit model
try:
    model = GPR(x_train, y_train, kernel, noise_sigma)
    model_error = None
except Exception as e:
    model, model_error = None, str(e)

# Title + Equation (always visible)
st.title("Single-Objective Gaussian Process (One-Step Posterior)")
st.latex(kernel_latex)

if model_error:
    st.error(f"Model error: {model_error}")
else:
    # Main plot
    st.plotly_chart(create_gpr_plot(model, x_pred, x_train, y_train, kernel_name_display), 
                    use_container_width=True)
    
    # Posterior samples
    st.markdown("---")
    col1, col2 = st.columns([6, 1])
    with col2:
        n_samples = st.slider("Samples", 1, 8, 5, key='n_samples')
    
    np.random.seed(st.session_state.get('seed', 42))
    st.plotly_chart(create_sample_plot(model, x_pred, x_train, y_train, n_samples), 
                    use_container_width=True)
    
    # Diagnostics
    st.markdown("---")
    st.subheader("Diagnostics")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        diag = model.get_diagnostics()
        st.metric("Training points", diag['n_train_points'])
        cond = diag['condition_number']
        st.metric("Condition number", f"{cond:.1e}" if cond < 1e12 else "Large")
        
        # Display variance from covariance matrix
        pred_var = model._memory['variance']
        st.metric("Mean pred. variance", f"{np.mean(pred_var):.4f}")
        st.metric("Max pred. variance", f"{np.max(pred_var):.4f}")
        
        if cond > 1e10:
            st.warning("Ill-conditioned. Try adding noise.")
    with col2:
        st.plotly_chart(create_kernel_heatmap(model), use_container_width=True)
