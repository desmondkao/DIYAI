import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
import numpy as np
import re
import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import altair as alt

# Constants for energy reporting - 2025 updated values
CO2_PER_KWH = {
    "global_avg": 0.442,  # kg CO2 per kWh (global average, 2025)
    "eu": 0.198,          # kg CO2 per kWh (EU average, 2025)
    "us": 0.382,          # kg CO2 per kWh (US average, 2025)
    "china": 0.529,       # kg CO2 per kWh (China average, 2025)
    "india": 0.682        # kg CO2 per kWh (India average, 2025)
}

# Based on recent research on cloud vs local processing efficiency
CLOUD_EFFICIENCY_FACTOR = 0.58  # Cloud efficiency factor (42% more efficient than local)
SPECIALIZED_HARDWARE_FACTOR = 0.75  # TPU/specialized hardware vs standard GPU

# AI model carbon intensity benchmarks (for comparison)
MODEL_CARBON_BENCHMARKS = {
    "large_llm_training": 250.0,    # Estimated tonnes of CO2 for training a large LLM
    "image_generation": 0.00085,    # kg CO2 per image generated
    "nlp_inference": 0.00032,       # kg CO2 per text generation (average)
    "mobile_optimized": 0.00006     # kg CO2 per inference on optimized mobile model
}

# Set page config
st.set_page_config(
    page_title="Sustainable Text Generation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to style the app
st.markdown("""
<style>
    .reportview-container {
        background-color: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    .stProgress .st-bo {
        background-color: #2ecc71;
    }
    .sustainability-metric {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .main-header {
        color: #2c3e50;
    }
    .good-metric {
        color: #2ecc71;
        font-weight: bold;
    }
    .bad-metric {
        color: #e74c3c;
        font-weight: bold;
    }
    .info-box {
        background-color: #ebf5fb;
        border-left: 5px solid #3498db;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        color: #7f8c8d;
        font-size: 0.8em;
        margin-top: 50px;
    }
</style>
""", unsafe_allow_html=True)

class EnergyTracker:
    def __init__(self, log_file="energy_logs.json"):
        self.log_file = log_file
        self.start_time = None
        self.end_time = None
        self.gpu_available = tf.config.list_physical_devices('GPU')
        self.logs = self._load_logs()
        self.session_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": 0,
            "energy_kwh": 0,
            "co2_emissions_kg": 0,
            "model_info": {},
            "hardware_info": self._get_hardware_info(),
            "comparison": {}
        }
    
    def _load_logs(self):
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    return json.load(f)
            except:
                return {"sessions": []}
        return {"sessions": []}
    
    def _get_hardware_info(self):
        info = {"device": "CPU"}
        if self.gpu_available:
            # Get GPU information if available
            gpu_info = tf.config.experimental.get_device_details(self.gpu_available[0])
            info = {
                "device": "GPU",
                "name": gpu_info.get('device_name', 'Unknown GPU'),
                "compute_capability": gpu_info.get('compute_capability', 'Unknown')
            }
        return info
    
    def _estimate_power_consumption(self, duration_seconds):
        """Estimate power consumption based on hardware type and utilization"""
        if self.gpu_available:
            # Rough estimate for GPU power consumption (W)
            avg_gpu_power = 150  # watts for a mid-range GPU
            return (avg_gpu_power * duration_seconds) / 3600000  # Convert to kWh
        else:
            # Estimate for CPU training
            avg_cpu_power = 65  # watts for CPU under load
            return (avg_cpu_power * duration_seconds) / 3600000  # Convert to kWh
    
    def start(self):
        """Start tracking energy consumption"""
        self.start_time = time.time()
        st.info(f"📊 Energy tracking started on {'GPU' if self.gpu_available else 'CPU'}")
    
    def stop(self, model=None):
        """Stop tracking and compute energy metrics"""
        if not self.start_time:
            st.warning("Warning: Energy tracking wasn't started")
            return
        
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        # Store model info if provided
        if model:
            self.session_data["model_info"] = {
                "type": type(model).__name__,
                "layers": len(model.layers),
                "parameters": model.count_params()
            }
        
        # Calculate energy consumption
        energy_kwh = self._estimate_power_consumption(duration)
        
        # Calculate CO2 emissions for different regions
        co2_emissions = {region: energy_kwh * factor for region, factor in CO2_PER_KWH.items()}
        
        # Calculate what cloud would use (estimated)
        cloud_energy_kwh = energy_kwh * CLOUD_EFFICIENCY_FACTOR
        cloud_co2 = {region: cloud_energy_kwh * factor for region, factor in CO2_PER_KWH.items()}
        
        # Update session data
        self.session_data.update({
            "duration_seconds": duration,
            "energy_kwh": energy_kwh,
            "co2_emissions_kg": co2_emissions,
            "comparison": {
                "cloud_energy_kwh": cloud_energy_kwh,
                "cloud_co2_emissions_kg": cloud_co2,
                "savings_percentage": (1 - CLOUD_EFFICIENCY_FACTOR) * 100
            }
        })
        
        # Save logs
        self.logs["sessions"].append(self.session_data)
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
        
        return self.session_data

# Function to preprocess and tokenize text
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])  # Fit tokenizer on text
    total_words = len(tokenizer.word_index) + 1  # Get total number of unique words

    token_list = tokenizer.texts_to_sequences([text])[0]  # Convert text to sequence of tokens
    max_sequence_len = 100  # Increase sequence length
    input_sequences = create_sequences(token_list, max_sequence_len)  # Create input sequences

    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))  # Pad sequences

    X, y = input_sequences[:, :-1], input_sequences[:, -1]  # Split input sequences into X and y
    y = tf.keras.utils.to_categorical(y, num_classes=total_words)  # One-hot encode y

    return tokenizer, total_words, max_sequence_len, X, y


def create_sequences(token_list, max_sequence_len):
    input_sequences = []
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        if len(n_gram_sequence) <= max_sequence_len:
            input_sequences.append(n_gram_sequence)
    return input_sequences


# Function to build the model with configurable size for sustainability
def build_model(total_words, max_sequence_len, model_size='medium'):
    # Define model configurations for different sizes
    sizes = {
        'small': {
            'embedding_dim': 100,
            'lstm_units': [64, 64],
            'dense_units': 64
        },
        'medium': {
            'embedding_dim': 200,
            'lstm_units': [128, 128],
            'dense_units': 128
        },
        'large': {
            'embedding_dim': 300,
            'lstm_units': [256, 256, 128],
            'dense_units': 256
        }
    }
    
    # Get configuration for the selected size
    config = sizes.get(model_size, sizes['medium'])
    
    model = Sequential()
    model.add(Embedding(input_dim=total_words, output_dim=config['embedding_dim']))
    
    # Add LSTM layers based on configuration
    for i, units in enumerate(config['lstm_units']):
        return_sequences = i < len(config['lstm_units']) - 1
        model.add(Bidirectional(LSTM(units, return_sequences=return_sequences)))
        model.add(Dropout(0.3))
    
    model.add(Dense(config['dense_units'], activation='relu'))
    model.add(Dense(total_words, activation='softmax'))
    
    # Use mixed precision for better efficiency on GPUs
    if tf.config.list_physical_devices('GPU'):
        st.info("Using mixed precision for better GPU efficiency")
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model


# Sampling methods for text generation
def top_k_sampling(predictions, k=10):
    sorted_indices = np.argsort(predictions)[::-1][:k]
    top_k_probabilities = predictions[sorted_indices]
    top_k_probabilities = top_k_probabilities / np.sum(top_k_probabilities)
    predicted_index = np.random.choice(sorted_indices, p=top_k_probabilities)
    return predicted_index


def top_p_sampling(predictions, p=0.9):
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_probabilities = predictions[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probabilities)
    
    indices_to_keep = sorted_indices[cumulative_probs <= p]
    probabilities_to_keep = sorted_probabilities[:len(indices_to_keep)]
    probabilities_to_keep = probabilities_to_keep / np.sum(probabilities_to_keep)
    
    predicted_index = np.random.choice(indices_to_keep, p=probabilities_to_keep)
    return predicted_index


def generate_text(model, tokenizer, seed_text, next_words, max_sequence_len, temperature=0.7, top_k=None, top_p=None):
    energy_tracker = EnergyTracker()
    energy_tracker.start()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    generated_text = seed_text
    for i in range(next_words):
        status_text.text(f"Generating word {i+1}/{next_words}...")
        progress_bar.progress((i + 1) / next_words)
        
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predictions = model.predict(token_list, verbose=0)[0]

        # Apply temperature
        predictions = np.log(predictions) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)

        # Apply sampling strategy
        if top_k is not None:
            predicted_index = top_k_sampling(predictions, k=top_k)
        elif top_p is not None:
            predicted_index = top_p_sampling(predictions, p=top_p)
        else:
            predicted_index = np.random.choice(len(predictions), p=predictions)

        predicted_word = tokenizer.index_word.get(predicted_index, '')
        if predicted_word:
            generated_text += " " + predicted_word
        else:
            break
    
    # Track energy used for generation
    metrics = energy_tracker.stop(model)
    progress_bar.empty()
    status_text.empty()
    
    return generated_text, metrics


# Improved file IO functions with progress reporting
def read_source_text_in_chunks(file_path, chunk_size=1024):
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("This is a sample text to get started with. You can add more text to train the model.")
    
    total_size = os.path.getsize(file_path)
    read_size = 0
    text = ""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with open(file_path, 'r', encoding='utf-8') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            read_size += len(chunk)
            progress = read_size/total_size
            progress_bar.progress(progress)
            status_text.text(f"Reading file: {progress*100:.1f}% complete")
            text += chunk
    
    progress_bar.empty()
    status_text.empty()
    st.success("File reading complete!")
    return text


def append_to_source_file(file_path, new_text):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(f"\n{new_text}")
    st.success("Text added successfully to corpus!")


# Function to compare and visualize different model sizes
def compare_model_sizes(source_text, epochs=10):
    sizes = ['small', 'medium', 'large']
    results = []
    
    tokenizer, total_words, max_sequence_len, X, y = preprocess_text(source_text)
    
    progress_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    for size in sizes:
        progress_placeholder.info(f"Training {size} model...")
        
        # Build and train model
        energy_tracker = EnergyTracker()
        energy_tracker.start()
        
        model = build_model(total_words, max_sequence_len, model_size=size)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        
        # Create a custom callback to update progress
        class StreamlitCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress_placeholder.info(f"Training {size} model: Epoch {epoch+1}/{epochs}")
        
        model.fit(X, y, epochs=epochs, batch_size=64, 
                 callbacks=[early_stopping, StreamlitCallback()], 
                 verbose=0)
        
        metrics = energy_tracker.stop(model)
        
        # Save results
        results.append({
            'size': size,
            'params': model.count_params(),
            'energy_kwh': metrics['energy_kwh'],
            'co2_g': metrics['co2_emissions_kg']['global_avg'] * 1000
        })
        
        # Update chart as we go
        df = pd.DataFrame(results)
        chart = create_comparison_chart(df)
        chart_placeholder.plotly_chart(chart, use_container_width=True)
    
    progress_placeholder.empty()
    
    return results, chart_placeholder

# Create comparison chart
def create_comparison_chart(df):
    fig = go.Figure()
    
    # Add energy consumption bars
    fig.add_trace(go.Bar(
        x=df['size'],
        y=df['energy_kwh'] * 1000,  # Convert to Wh
        name='Energy (Wh)',
        marker_color='#3498db',
        text=df['energy_kwh'] * 1000,
        textposition='auto',
    ))
    
    # Add CO2 emissions bars
    fig.add_trace(go.Bar(
        x=df['size'],
        y=df['co2_g'],
        name='CO2 (g)',
        marker_color='#e74c3c',
        text=df['co2_g'],
        textposition='auto',
        visible='legendonly'  # Hide by default
    ))
    
    # Add parameters line on secondary axis
    fig.add_trace(go.Scatter(
        x=df['size'],
        y=df['params'],
        name='Parameters',
        mode='lines+markers',
        marker=dict(color='#2ecc71'),
        text=df['params'],
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        title='Model Size Comparison: Energy vs Parameters',
        xaxis_title='Model Size',
        yaxis_title='Energy Consumption (Wh)',
        yaxis2=dict(
            title='Number of Parameters',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Dashboard visualizations
def display_energy_metrics(metrics):
    st.subheader("Sustainability Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="⏱️ Processing Time",
            value=f"{metrics['duration_seconds']:.2f} s"
        )
    
    with col2:
        st.metric(
            label="⚡ Energy Consumption",
            value=f"{metrics['energy_kwh']*1000:.2f} Wh"
        )
    
    with col3:
        st.metric(
            label="🏭 CO2 Emissions",
            value=f"{metrics['co2_emissions_kg']['global_avg']*1000:.2f} g",
            delta=f"{(1-CLOUD_EFFICIENCY_FACTOR)*100:.0f}% reduction possible",
            delta_color="inverse"
        )
    
    # Create more detailed metrics
    st.subheader("Detailed Sustainability Analysis")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Cloud Comparison", "Regional Impact", "Reference Benchmarks"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Cloud vs local comparison
            cloud_data = pd.DataFrame({
                'Deployment': ['Local', 'Cloud'],
                'Energy (Wh)': [
                    metrics['energy_kwh'] * 1000,
                    metrics['comparison']['cloud_energy_kwh'] * 1000
                ],
                'CO2 (g)': [
                    metrics['co2_emissions_kg']['global_avg'] * 1000,
                    metrics['comparison']['cloud_co2_emissions_kg']['global_avg'] * 1000
                ]
            })
            
            fig = px.bar(
                cloud_data, 
                x='Deployment', 
                y=['Energy (Wh)', 'CO2 (g)'],
                barmode='group',
                title='Cloud vs. Local Processing',
                color_discrete_sequence=['#3498db', '#e74c3c']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Create gauge for cloud savings
            savings = metrics['comparison']['savings_percentage']
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=savings,
                title={'text': "Potential Cloud Savings (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#2ecc71"},
                    'steps': [
                        {'range': [0, 30], 'color': "#f39c12"},
                        {'range': [30, 70], 'color': "#2ecc71"},
                        {'range': [70, 100], 'color': "#27ae60"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': savings
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Regional impact
        region_data = pd.DataFrame({
            'Region': list(metrics['co2_emissions_kg'].keys()),
            'CO2 Emissions (g)': [v * 1000 for v in metrics['co2_emissions_kg'].values()]
        })
        
        fig = px.bar(
            region_data,
            x='Region',
            y='CO2 Emissions (g)',
            title='CO2 Emissions by Region',
            color='CO2 Emissions (g)',
            color_continuous_scale='Viridis'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Min-max region comparison
        min_region = min(metrics['co2_emissions_kg'].items(), key=lambda x: x[1])
        max_region = max(metrics['co2_emissions_kg'].items(), key=lambda x: x[1])
        diff_percentage = (max_region[1] - min_region[1]) / min_region[1] * 100
        
        st.info(f"By choosing {min_region[0]} over {max_region[0]} for deployment, you could reduce emissions by {diff_percentage:.1f}%")
    
    with tab3:
        if 'model_info' in metrics and 'parameters' in metrics['model_info']:
            # Model efficiency metrics
            model_size = metrics['model_info']['parameters']
            energy_per_param = metrics['energy_kwh']*1000/model_size*1000000
            co2_per_param = metrics['co2_emissions_kg']['global_avg']*1000/model_size*1000000
            
            st.write("##### Model Efficiency")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Parameters", f"{model_size:,}")
            with col2:
                st.metric("Energy per Million Parameters", f"{energy_per_param:.2f} Wh")
        
        # Create comparison to benchmarks
        one_image_gen_energy = MODEL_CARBON_BENCHMARKS['image_generation'] / CO2_PER_KWH['global_avg'] * 1000
        equivalent_images = metrics['energy_kwh']*1000/one_image_gen_energy
        llm_percentage = metrics['co2_emissions_kg']['global_avg']/MODEL_CARBON_BENCHMARKS['large_llm_training']*100
        
        st.write("##### Equivalent Environmental Impact")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Equivalent to generating", f"{equivalent_images:.1f} AI images")
        
        with col2:
            st.metric("% of large LLM training", f"{llm_percentage:.5f}%")
        
        # Create a chart showing model in context of other AI tasks
        benchmark_data = pd.DataFrame({
            'Task': ['Your Model', 'Image Generation', 'NLP Inference', 'Mobile-optimized'],
            'CO2 (g)': [
                metrics['co2_emissions_kg']['global_avg'] * 1000,
                MODEL_CARBON_BENCHMARKS['image_generation'] * 1000,
                MODEL_CARBON_BENCHMARKS['nlp_inference'] * 1000,
                MODEL_CARBON_BENCHMARKS['mobile_optimized'] * 1000
            ]
        })
        
        fig = px.bar(
            benchmark_data,
            x='Task',
            y='CO2 (g)',
            title='CO2 Emissions Compared to Other AI Tasks',
            color='CO2 (g)',
            log_y=True,  # Using log scale due to potential large differences
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)


def display_sustainability_report(log_file="energy_logs.json"):
    if not os.path.exists(log_file):
        st.warning("No energy logs found. Run some operations first to generate logs.")
        return
    
    try:
        with open(log_file, 'r') as f:
            logs = json.load(f)
    except:
        st.error("Error reading log file.")
        return
    
    if not logs.get("sessions", []):
        st.warning("No sessions recorded in logs.")
        return
    
    # Create dataframe from session data
    data = []
    for session in logs["sessions"]:
        row = {
            "timestamp": session["timestamp"],
            "duration_seconds": session["duration_seconds"],
            "energy_kwh": session["energy_kwh"],
            "co2_global": session["co2_emissions_kg"]["global_avg"],
            "cloud_energy_kwh": session.get("comparison", {}).get("cloud_energy_kwh", 0),
            "cloud_co2_global": session.get("comparison", {}).get("cloud_co2_emissions_kg", {}).get("global_avg", 0),
            "hardware_type": session.get("hardware_info", {}).get("device", "Unknown")
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Calculate summary statistics
    total_energy = df["energy_kwh"].sum()
    total_co2 = df["co2_global"].sum()
    total_cloud_energy = df["cloud_energy_kwh"].sum()
    total_cloud_co2 = df["cloud_co2_global"].sum()
    
    # Calculate sustainability score
    gpu_sessions = df[df["hardware_type"] == "GPU"]
    gpu_percentage = len(gpu_sessions) / len(df) if len(df) > 0 else 0
    cloud_savings = (1 - total_cloud_energy/total_energy)*100 if total_energy > 0 else 0
    
    sustainability_score = 0
    if total_co2 < 0.1:  # Less than 100g CO2
        sustainability_score += 40
    elif total_co2 < 1:  # Less than 1kg CO2
        sustainability_score += 25
    elif total_co2 < 10:  # Less than 10kg CO2
        sustainability_score += 10
        
    if cloud_savings > 40:
        sustainability_score += 30
    elif cloud_savings > 20:
        sustainability_score += 15
        
    if gpu_percentage < 0.3:
        sustainability_score += 30
    elif gpu_percentage < 0.7:
        sustainability_score += 15
    
    # Determine score category
    score_category = "red"
    if sustainability_score >= 70:
        score_category = "green"
    elif sustainability_score >= 40:
        score_category = "orange"
    
    # Display summary dashboard
    st.header("Sustainability Report")
    
    # Score gauge
    score_color = "#e74c3c"  # red
    if score_category == "green":
        score_color = "#2ecc71"
    elif score_category == "orange":
        score_color = "#f39c12"
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sustainability_score,
            title={'text': "Sustainability Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': score_color},
                'steps': [
                    {'range': [0, 40], 'color': "#f5f5f5"},
                    {'range': [40, 70], 'color': "#f5f5f5"},
                    {'range': [70, 100], 'color': "#f5f5f5"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': sustainability_score
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Detailed metrics
        st.subheader("Total Environmental Impact")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric(
                label="Energy Consumption", 
                value=f"{total_energy*1000:.2f} Wh"
            )
            
        with metric_col2:
            st.metric(
                label="CO2 Emissions", 
                value=f"{total_co2*1000:.2f} g CO2e"
            )
            
        with metric_col3:
            st.metric(
                label="Training Sessions", 
                value=f"{len(df)}"
            )
        
        # Energy over time chart
        st.subheader("Energy Consumption Over Time")
        
        # Convert timestamp to datetime for chart
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('datetime')
        
        # Create chart
        fig = go.Figure()
        
        # Add energy consumption line
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['energy_kwh'] * 1000,  # Convert to Wh
            mode='lines+markers',
            name='Energy (Wh)',
            line=dict(color='#3498db', width=2)
        ))
        
        # Add cloud energy consumption line
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['cloud_energy_kwh'] * 1000,  # Convert to Wh
            mode='lines+markers',
            name='Cloud Energy (Wh)',
            line=dict(color='#2ecc71', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Energy Consumption History',
            xaxis_title='Time',
            yaxis_title='Energy (Wh)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add tabs for more detailed analysis
        detailed_tab1, detailed_tab2, detailed_tab3 = st.tabs([
            "Hardware Analysis", 
            "Regional Impact", 
            "Best Practices"
        ])
        
        with detailed_tab1:
            # Hardware distribution pie chart
            hardware_counts = df['hardware_type'].value_counts().reset_index()
            hardware_counts.columns = ['Hardware', 'Count']
            
            fig = px.pie(
                hardware_counts, 
                values='Count', 
                names='Hardware',
                title='Hardware Distribution',
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Hardware efficiency comparison
            st.subheader("Hardware Efficiency Comparison")
            
            # Compare CPU vs GPU vs specialized hardware
            hw_compare_data = pd.DataFrame({
                'Hardware': ['CPU', 'GPU', 'Specialized AI Hardware'],
                'Relative Energy': [1.0, 0.8, 0.6],  # Normalized values
                'Description': [
                    'Baseline energy consumption',
                    'More efficient for AI workloads',
                    'Optimized for neural networks (TPU/ASIC)'
                ]
            })
            
            # Show as horizontal bar chart
            fig = px.bar(
                hw_compare_data,
                y='Hardware',
                x='Relative Energy',
                orientation='h',
                color='Relative Energy',
                color_continuous_scale='Viridis_r',  # Reversed scale
                labels={'Relative Energy': 'Relative Energy Consumption'},
                text='Description'
            )
            
            fig.update_traces(textposition='inside')
            fig.update_layout(height=300)
            
            st.plotly_chart(fig, use_container_width=True)
            
        with detailed_tab2:
            # Regional impact comparison
            st.subheader("CO2 Intensity by Region")
            
            # Create data for regional CO2 intensity
            region_intensity = pd.DataFrame({
                'Region': list(CO2_PER_KWH.keys()),
                'CO2 Intensity (kg/kWh)': list(CO2_PER_KWH.values())
            })
            
            # Show as horizontal bar chart
            fig = px.bar(
                region_intensity,
                y='Region',
                x='CO2 Intensity (kg/kWh)',
                orientation='h',
                color='CO2 Intensity (kg/kWh)',
                color_continuous_scale='Viridis',
                title='CO2 Intensity by Region (2025)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show potential savings by region
            st.subheader("Potential CO2 Savings by Region")
            
            # Calculate savings compared to global average
            global_avg = CO2_PER_KWH['global_avg']
            savings_data = []
            
            for region, intensity in CO2_PER_KWH.items():
                if region != 'global_avg':
                    savings_pct = ((global_avg - intensity) / global_avg) * 100
                    savings_data.append({
                        'Region': region,
                        'Savings vs Global Avg (%)': savings_pct
                    })
            
            savings_df = pd.DataFrame(savings_data)
            
            # Show as horizontal bar chart
            fig = px.bar(
                savings_df,
                y='Region',
                x='Savings vs Global Avg (%)',
                orientation='h',
                color='Savings vs Global Avg (%)',
                color_continuous_scale='RdYlGn',
                title='CO2 Savings vs Global Average'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        with detailed_tab3:
            st.subheader("2025 Sustainability Best Practices")
            
            # Create tabs for different categories
            bp_tab1, bp_tab2, bp_tab3 = st.tabs([
                "Hardware Optimization", 
                "Model Efficiency", 
                "Deployment Strategies"
            ])
            
            with bp_tab1:
                st.markdown("""
                #### Hardware Optimization
                - **Use specialized AI hardware** like TPUs which can reduce energy consumption by up to 25% compared to standard GPUs
                - **Batch processing** to maximize hardware utilization and reduce overall energy use
                - **Mixed precision training** to reduce memory and computation requirements
                - **Memory-efficient attention mechanisms** for transformer models
                """)
                
                # Create visual for hardware optimization impact
                impact_data = pd.DataFrame({
                    'Optimization': [
                        'Specialized Hardware',
                        'Batched Processing',
                        'Mixed Precision',
                        'Memory Optimization'
                    ],
                    'Energy Reduction (%)': [25, 18, 15, 12]
                })
                
                fig = px.bar(
                    impact_data,
                    x='Optimization',
                    y='Energy Reduction (%)',
                    color='Energy Reduction (%)',
                    color_continuous_scale='Viridis',
                    title='Impact of Hardware Optimizations'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            with bp_tab2:
                st.markdown("""
                #### Model Efficiency Techniques
                - **Knowledge distillation**: Train smaller, more efficient models from larger ones to reduce inference energy by 60-95%
                - **Quantization**: Apply post-training quantization to reduce model size and computation requirements by 75%
                - **Pruning**: Remove redundant connections and neurons without significant accuracy loss
                - **Sparse activation**: Activate only a small portion of the network for each input
                """)
                
                # Create visual for model optimization impact
                impact_data = pd.DataFrame({
                    'Technique': [
                        'Knowledge Distillation',
                        'Quantization',
                        'Pruning',
                        'Sparse Activation'
                    ],
                    'Model Size Reduction (%)': [90, 75, 60, 80],
                    'Performance Impact (%)': [5, 2, 3, 5]
                })
                
                # Create bubble chart
                fig = px.scatter(
                    impact_data,
                    x='Performance Impact (%)',
                    y='Model Size Reduction (%)',
                    size='Model Size Reduction (%)',
                    color='Performance Impact (%)',
                    text='Technique',
                    color_continuous_scale='Viridis_r',  # Reversed scale (lower is better)
                    title='Model Efficiency Techniques: Size Reduction vs Performance Impact'
                )
                
                fig.update_traces(textposition='top center')
                
                st.plotly_chart(fig, use_container_width=True)
                
            with bp_tab3:
                st.markdown("""
                #### Deployment Strategies
                - **Regional selection**: Deploy models in regions with low-carbon electricity grids
                - **Carbon-aware scheduling**: Schedule training jobs during times when electricity has lower carbon intensity
                - **Edge deployment**: Deploy smaller models on edge devices to reduce data center dependence
                - **Cloud optimization**: Use auto-scaling and right-sizing for cloud resources
                """)
                
                # Create visual for deployment strategies
                strategies_data = pd.DataFrame({
                    'Strategy': [
                        'Regional Selection',
                        'Carbon-aware Scheduling',
                        'Edge Deployment',
                        'Cloud Optimization'
                    ],
                    'Implementation Difficulty': [1, 2, 3, 2],  # 1=Easy, 3=Hard
                    'CO2 Reduction (%)': [70, 35, 25, 30]
                })
                
                fig = px.scatter(
                    strategies_data,
                    x='Implementation Difficulty',
                    y='CO2 Reduction (%)',
                    color='CO2 Reduction (%)',
                    size='CO2 Reduction (%)',
                    text='Strategy',
                    color_continuous_scale='Viridis',
                    labels={'Implementation Difficulty': 'Implementation Difficulty (1=Easy, 3=Hard)'}
                )
                
                fig.update_traces(textposition='top center')
                fig.update_xaxes(tickvals=[1, 2, 3])
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Display CSV download button
    st.subheader("Export Data")
    
    # Convert data to CSV
    csv = df.to_csv(index=False)
    
    # Add download button
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name="sustainability_metrics.csv",
        mime="text/csv"
    )
    
    # Add Capgemini branding footer
    st.markdown("""
    <div class="footer">
        <p>Sustainable AI Model Report</p>
        <p>Generated for Sustainability Analysis - 2025</p>
    </div>
    """, unsafe_allow_html=True)


# Main Streamlit app
def main():
    # App title and description
    st.sidebar.title("Sustainable Text Generation")
    st.sidebar.info(
        "This application demonstrates a sustainable approach to text generation "
        "with real-time energy and carbon footprint monitoring."
    )
    
    # Main navigation
    st.sidebar.header("Navigation")
    pages = [
        "Dashboard Overview",
        "Train Model",
        "Generate Text",
        "Add Text to Corpus",
        "Compare Model Sizes",
        "Sustainability Report"
    ]
    
    page = st.sidebar.radio("Go to", pages)
    
    # File paths
    source_file = 'source_text.txt'
    model_path = 'trained_model.h5'
    
    # Page content
    if page == "Dashboard Overview":
        st.title("EcoLLM: Visualizing the Environmental Cost of AI")
        
        st.markdown("""
        <div class="info-box">
            <h3>Welcome to the Sustainable Text Generation Platform</h3>
            <p>This application demonstrates how AI text generation can be implemented with sustainability in mind,
            tracking and optimizing energy usage and carbon emissions in real-time.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display key features
        st.subheader("Key Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### Sustainability Metrics
            - Real-time energy tracking during model operations
            - Carbon emissions calculation based on regional grids
            - Cloud vs. local comparison for optimization
            - Hardware efficiency recommendations
            """)
            
        with col2:
            st.markdown("""
            #### Text Generation
            - Configurable model sizes for efficiency
            - Multiple sampling techniques
            - Interactive text generation
            - Energy-efficiency metrics for each generation
            """)
        
        # Display system status
        st.subheader("System Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Check if model exists
            model_exists = os.path.exists(model_path)
            st.metric("Model Status", "Ready" if model_exists else "Not Trained")
            
        with col2:
            # Check for GPU
            gpus = tf.config.list_physical_devices('GPU')
            st.metric("Hardware", "GPU Available" if gpus else "CPU Only")
            
        with col3:
            # Check for energy logs
            logs_exist = os.path.exists("energy_logs.json")
            st.metric("Energy Logs", "Available" if logs_exist else "Not Available")
        
        # Check if we have energy logs to display some charts
        if logs_exist:
            try:
                with open("energy_logs.json", 'r') as f:
                    logs = json.load(f)
                
                if logs.get("sessions", []):
                    st.subheader("Recent Energy Usage")
                    
                    # Create dataframe from last 5 sessions
                    data = []
                    for session in logs["sessions"][-5:]:
                        row = {
                            "timestamp": session["timestamp"],
                            "duration_seconds": session["duration_seconds"],
                            "energy_kwh": session["energy_kwh"],
                            "co2_global": session["co2_emissions_kg"]["global_avg"]
                        }
                        data.append(row)
                    
                    df = pd.DataFrame(data)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp')
                    
                    # Create chart
                    fig = px.line(
                        df,
                        x='timestamp',
                        y=['energy_kwh', 'co2_global'],
                        title='Recent Energy and CO2 Metrics',
                        labels={
                            'energy_kwh': 'Energy (kWh)',
                            'co2_global': 'CO2 (kg)',
                            'timestamp': 'Time'
                        }
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            except:
                pass
        
        # Quick start guide
        st.subheader("Quick Start Guide")
        
        st.markdown("""
        1. **Add Text to Corpus**: Start by adding text to train the model
        2. **Train Model**: Train the text generation model with your corpus
        3. **Generate Text**: Generate new text based on a seed phrase
        4. **Compare Models**: Compare energy efficiency of different model sizes
        5. **View Report**: Generate a detailed sustainability report
        """)
        
    elif page == "Train Model":
        st.title("Train Model")
        
        # Check if source file exists and has content
        if not os.path.exists(source_file) or os.path.getsize(source_file) == 0:
            st.warning("No training data found. Please add text to the corpus first.")
            st.stop()
        
        # Read source text
        source_text = read_source_text_in_chunks(source_file)
        
        # Display text statistics
        st.subheader("Corpus Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Text Length", f"{len(source_text)} chars")
        
        with col2:
            words = len(source_text.split())
            st.metric("Word Count", f"{words} words")
        
        with col3:
            lines = len(source_text.splitlines())
            st.metric("Line Count", f"{lines} lines")
        
        # Training form
        st.subheader("Training Configuration")
        
        with st.form("train_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                model_size = st.selectbox(
                    "Model Size",
                    options=["small", "medium", "large"],
                    index=1,  # Default to medium
                    help="Larger models are more capable but consume more energy"
                )
            
            with col2:
                epochs = st.slider(
                    "Training Epochs",
                    min_value=5,
                    max_value=100,
                    value=50,
                    step=5,
                    help="More epochs may improve model quality but consume more energy"
                )
            
            train_button = st.form_submit_button("Train Model")
        
        if train_button:
            # Set up progress display
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Energy tracker
            energy_tracker = EnergyTracker()
            energy_tracker.start()
            
            # Preprocess text
            status_text.text("Preprocessing text...")
            tokenizer, total_words, max_sequence_len, X, y = preprocess_text(source_text)
            
            # Build model
            status_text.text("Building model...")
            progress_bar.progress(10)
            model = build_model(total_words, max_sequence_len, model_size=model_size)
            
            # Create custom callback for progress updates
            class ProgressCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress = 10 + int(90 * (epoch + 1) / epochs)
                    progress_bar.progress(min(progress, 100))
                    status_text.text(f"Training epoch {epoch+1}/{epochs}...")
            
            # Train model
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=5,
                restore_best_weights=True
            )
            
            model.fit(
                X, y,
                epochs=epochs,
                batch_size=64,
                callbacks=[early_stopping, ProgressCallback()],
                verbose=0
            )
            
            # Save model
            status_text.text("Saving model...")
            model.save(model_path)
            
            # Save metadata
            with open('model_metadata.json', 'w') as f:
                json.dump({
                    'total_words': total_words,
                    'max_sequence_len': max_sequence_len,
                    'model_size': model_size,
                    'date_trained': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }, f)
            
            # Get energy metrics
            metrics = energy_tracker.stop(model)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display success message
            st.success(f"Model trained and saved successfully!")
            
            # Display energy metrics
            display_energy_metrics(metrics)
        
        # Display existing model info if available
        if os.path.exists(model_path) and os.path.exists('model_metadata.json'):
            st.subheader("Current Model Information")
            
            try:
                with open('model_metadata.json', 'r') as f:
                    metadata = json.load(f)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Model Size", metadata.get('model_size', 'Unknown'))
                
                with col2:
                    st.metric("Vocabulary Size", metadata.get('total_words', 'Unknown'))
                
                with col3:
                    st.metric("Date Trained", metadata.get('date_trained', 'Unknown'))
            except:
                st.error("Error loading model metadata")
    
    elif page == "Generate Text":
        st.title("Generate Text")
        
        # Check if model exists
        if not os.path.exists(model_path):
            st.warning("No trained model found. Please train a model first.")
            st.stop()
        
        # Check if source file exists
        if not os.path.exists(source_file):
            st.warning("No source text found. Please add text to the corpus first.")
            st.stop()
        
        # Load model and metadata
        model = load_model(model_path)
        
        try:
            with open('model_metadata.json', 'r') as f:
                metadata = json.load(f)
        except:
            st.error("Error loading model metadata")
            st.stop()
        
        # Read source text to get tokenizer
        source_text = read_source_text_in_chunks(source_file)
        tokenizer, total_words, max_sequence_len, _, _ = preprocess_text(source_text)
        
        # Display model info
        st.subheader("Model Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Size", metadata.get('model_size', 'Unknown'))
        
        with col2:
            st.metric("Vocabulary Size", metadata.get('total_words', 'Unknown'))
        
        with col3:
            st.metric("Sequence Length", metadata.get('max_sequence_len', 'Unknown'))
        
        # Text generation form
        st.subheader("Text Generation")
        
        with st.form("generation_form"):
            seed_text = st.text_area(
                "Seed Text",
                value="The sustainable approach to AI development",
                height=100,
                help="Starting text for generation"
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                next_words = st.slider(
                    "Number of Words",
                    min_value=10,
                    max_value=200,
                    value=50,
                    step=10,
                    help="Number of words to generate"
                )
            
            with col2:
                temperature = st.slider(
                    "Temperature",
                    min_value=0.1,
                    max_value=1.5,
                    value=0.7,
                    step=0.1,
                    help="Higher values make the text more creative but less coherent"
                )
            
            with col3:
                sampling_method = st.selectbox(
                    "Sampling Method",
                    options=["Default", "Top-K", "Top-P"],
                    index=0,
                    help="Method used to sample the next word"
                )
            
            # Show additional parameters based on sampling method
            if sampling_method == "Top-K":
                top_k = st.slider(
                    "K Value",
                    min_value=1,
                    max_value=50,
                    value=10,
                    help="Number of most likely next words to consider"
                )
            elif sampling_method == "Top-P":
                top_p = st.slider(
                    "P Value",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.9,
                    step=0.05,
                    help="Cumulative probability threshold for next word candidates"
                )
            
            generate_button = st.form_submit_button("Generate Text")
        
        if generate_button:
            # Generate text based on sampling method
            if sampling_method == "Top-K":
                generated_text, metrics = generate_text(
                    model, tokenizer, seed_text, next_words, max_sequence_len,
                    temperature=temperature, top_k=top_k
                )
            elif sampling_method == "Top-P":
                generated_text, metrics = generate_text(
                    model, tokenizer, seed_text, next_words, max_sequence_len,
                    temperature=temperature, top_p=top_p
                )
            else:
                generated_text, metrics = generate_text(
                    model, tokenizer, seed_text, next_words, max_sequence_len,
                    temperature=temperature
                )
            
            # Display generated text
            st.subheader("Generated Text")
            
            st.markdown(
                f"""<div style="background-color: #f8f9fa; border-radius: 5px; padding: 15px; border: 1px solid #e9ecef;">
                {generated_text}
                </div>""",
                unsafe_allow_html=True
            )
            
            # Display energy metrics
            display_energy_metrics(metrics)
    
    elif page == "Add Text to Corpus":
        st.title("Add Text to Corpus")
        
        # Show current corpus stats if exists
        if os.path.exists(source_file):
            source_text = read_source_text_in_chunks(source_file)
            
            st.subheader("Current Corpus Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Text Length", f"{len(source_text)} chars")
            
            with col2:
                words = len(source_text.split())
                st.metric("Word Count", f"{words} words")
            
            with col3:
                lines = len(source_text.splitlines())
                st.metric("Line Count", f"{lines} lines")
            
            # Show sample of current text
            with st.expander("View Sample of Current Text"):
                st.text(source_text[:500] + "..." if len(source_text) > 500 else source_text)
        
        # Form to add new text
        st.subheader("Add New Text")
        
        with st.form("add_text_form"):
            new_text = st.text_area(
                "Enter Text to Add",
                height=300,
                help="This text will be added to your training corpus"
            )
            
            add_button = st.form_submit_button("Add to Corpus")
        
        if add_button and new_text:
            # Add text to corpus
            append_to_source_file(source_file, new_text)
            
            # Update displayed stats
            source_text = read_source_text_in_chunks(source_file)
            
            st.subheader("Updated Corpus Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Text Length", f"{len(source_text)} chars")
            
            with col2:
                words = len(source_text.split())
                st.metric("Word Count", f"{words} words")
            
            with col3:
                lines = len(source_text.splitlines())
                st.metric("Line Count", f"{lines} lines")
        
        # Option to upload a file
        st.subheader("Or Upload a Text File")
        
        uploaded_file = st.file_uploader("Choose a text file", type=["txt"])
        
        if uploaded_file is not None:
            # Read file
            uploaded_text = uploaded_file.read().decode("utf-8")
            
            # Show preview
            st.subheader("File Preview")
            st.text(uploaded_text[:500] + "..." if len(uploaded_text) > 500 else uploaded_text)
            
            # Add button
            if st.button("Add File to Corpus"):
                append_to_source_file(source_file, uploaded_text)
                st.success("File content added to corpus!")
                
                # Update displayed stats
                source_text = read_source_text_in_chunks(source_file)
                
                st.subheader("Updated Corpus Statistics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Text Length", f"{len(source_text)} chars")
                
                with col2:
                    words = len(source_text.split())
                    st.metric("Word Count", f"{words} words")
                
                with col3:
                    lines = len(source_text.splitlines())
                    st.metric("Line Count", f"{lines} lines")
    
    elif page == "Compare Model Sizes":
        st.title("Compare Model Sizes")
        
        # Check if source file exists
        if not os.path.exists(source_file):
            st.warning("No source text found. Please add text to the corpus first.")
            st.stop()
        
        st.markdown("""
        <div class="info-box">
            <p>This tool helps you compare the environmental impact of different model sizes.
            It trains small, medium, and large models on your corpus and measures their energy consumption and carbon footprint.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Configuration form
        with st.form("compare_form"):
            epochs = st.slider(
                "Training Epochs per Model",
                min_value=2,
                max_value=20,
                value=5,
                help="Higher values give more accurate comparison but take longer"
            )
            
            compare_button = st.form_submit_button("Compare Models")
        
        if compare_button:
            # Read source text
            source_text = read_source_text_in_chunks(source_file)
            
            # Run comparison
            results, chart_placeholder = compare_model_sizes(source_text, epochs=epochs)
            
            # Display results in a table
            st.subheader("Comparison Results")
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            
            # Download button for results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="model_comparison.csv",
                mime="text/csv"
            )
    
    elif page == "Sustainability Report":
        st.title("Sustainability Report")
        
        if st.button("Generate Sustainability Report"):
            display_sustainability_report()


if __name__ == "__main__":
    # Configure TensorFlow for better efficiency
    tf.config.optimizer.set_jit(True)  # Enable XLA optimization
    
    # Print available hardware
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        st.sidebar.success(f"Running with GPU: {gpus}")
        # Set memory growth to avoid allocating all memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        st.sidebar.info("Running with CPU only")
    
    main()