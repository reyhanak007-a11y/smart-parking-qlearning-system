import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import math
import time
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
from dotenv import load_dotenv
from utils.parking_env import ParkingEnvironment

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default-secret-key')

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
STATIC_IMG_DIR = os.path.join(BASE_DIR, 'static', 'img')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_IMG_DIR, exist_ok=True)

# Global variables
q_table = None
training_history = []
env = ParkingEnvironment()

@app.route('/')
def index():
    """Halaman utama aplikasi"""
    return render_template('index.html')

@app.route('/train', methods=['GET', 'POST'])
def train():
    """Halaman untuk melatih model Q-Learning"""
    global q_table, training_history
    
    if request.method == 'POST':
        try:
            # Get training parameters from form
            episodes = int(request.form.get('episodes', 1000))
            alpha = float(request.form.get('alpha', 0.9))
            gamma = float(request.form.get('gamma', 0.9))
            epsilon = float(request.form.get('epsilon', 1.0))
            min_epsilon = float(request.form.get('min_epsilon', 0.01))
            decay_rate = float(request.form.get('decay_rate', 0.005))
            
            # Reset environment and train model
            training_history = []
            env.reset()
            
            # Train the model
            q_table, training_history = q_learning_train(
                env=env,
                episodes=episodes,
                alpha=alpha,
                gamma=gamma,
                epsilon=epsilon,
                min_epsilon=min_epsilon,
                decay_rate=decay_rate
            )
            
            # Save Q-Table
            model_path = os.path.join(MODEL_DIR, 'q_table.npy')
            np.save(model_path, q_table)
            
            # Generate learning curve
            plot_filename = plot_learning_curve(training_history)
            
            # Return success message
            return jsonify({
                'success': True,
                'message': f'Model berhasil dilatih dengan {episodes} episode!',
                'plot_url': plot_filename,
                'avg_final_reward': np.mean(training_history[-100:]) if len(training_history) > 100 else training_history[-1]
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Terjadi error saat pelatihan: {str(e)}'
            })
    
    # GET request - show training page
    return render_template('training.html')

@app.route('/policy')
def policy():
    """Halaman untuk menampilkan kebijakan optimal"""
    global q_table, env
    
    if q_table is None:
        # Try to load pre-trained model
        model_path = os.path.join(MODEL_DIR, 'q_table.npy')
        if os.path.exists(model_path):
            q_table = np.load(model_path)
        else:
            return render_template('policy.html', error="Model belum dilatih. Silakan latih model terlebih dahulu.")
    
    # Generate policy visualization
    plot_filename = visualize_policy(env, q_table)
    
    # Get current slot status for display
    env.reset()
    slot_status = env.slot_status
    
    return render_template('policy.html', plot_path=plot_filename, slot_status=slot_status)

@app.route('/demo')
def demo():
    """Halaman untuk mendemonstrasikan kebijakan optimal"""
    global q_table, env
    
    if q_table is None:
        # Try to load pre-trained model
        model_path = os.path.join(MODEL_DIR, 'q_table.npy')
        if os.path.exists(model_path):
            q_table = np.load(model_path)
        else:
            return render_template('demo.html', error="Model belum dilatih. Silakan latih model terlebih dahulu.")
    
    # Run demo and get results
    env.reset()
    demo_results = run_demo(env, q_table)
    
    return render_template('demo.html', 
                          grid=demo_results['grid'],
                          steps=demo_results['steps'],
                          total_reward=demo_results['total_reward'],
                          history=demo_results['history'])

@app.route('/hyperparam_experiment')
def hyperparam_experiment():
    """Halaman untuk menampilkan hasil eksperimen hiperparameter"""
    # Generate comparison plot
    plot_filename = run_hyperparameter_experiment()
    
    return render_template('hyperparam.html', plot_path=plot_filename)

@app.route('/static/img/<path:filename>')
def serve_image(filename):
    """Serve static images"""
    return send_from_directory(STATIC_IMG_DIR, filename)

def q_learning_train(env, episodes=1000, alpha=0.9, gamma=0.9, epsilon=1.0, min_epsilon=0.01, decay_rate=0.005):
    """Train Q-Learning model and return Q-table and rewards history"""
    state_size = env.grid_size * env.grid_size * (2 ** len(env.parking_slots))
    action_size = len(env.action_space)
    q_table = np.zeros((state_size, action_size))
    
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        max_steps = 30
        
        while not done and step < max_steps:
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action = random.choice(env.action_space)  # Exploration
            else:
                action = np.argmax(q_table[state])  # Exploitation
            
            # Take action
            new_state, reward, done = env.step(action)
            
            # Update Q-table
            old_value = q_table[state, action]
            next_max = np.max(q_table[new_state])
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[state, action] = new_value
            
            # Update state and reward
            state = new_state
            total_reward += reward
            step += 1
        
        # Record rewards and decay epsilon
        rewards_history.append(total_reward)
        epsilon = max(min_epsilon, epsilon * math.exp(-decay_rate * episode))
    
    return q_table, rewards_history

def plot_learning_curve(rewards_history, window_size=100):
    """Generate and save learning curve plot"""
    rewards_series = pd.Series(rewards_history)
    rolling_mean = rewards_series.rolling(window=window_size, min_periods=1).mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_history, alpha=0.3, color='blue', label='Reward per Episode')
    plt.plot(rolling_mean, color='green', linewidth=2, label=f'Moving Average ({window_size} episode)')
    plt.title('Kurva Pembelajaran Q-Learning untuk Sistem Parkir Otomatis', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    timestamp = int(time.time())
    plot_filename = f"learning_curve_{timestamp}.png"
    plot_path = os.path.join(STATIC_IMG_DIR, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    
    return plot_filename

def visualize_policy(env, q_table):
    """Generate and save policy visualization"""
    # Reset environment to get consistent state
    env.reset()
    slot_status = env.slot_status.copy()
    
    # Create grid for visualization
    grid = np.zeros((env.grid_size, env.grid_size), dtype=int)
    action_grid = [['' for _ in range(env.grid_size)] for _ in range(env.grid_size)]
    
    # Generate optimal actions for each position
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            # Calculate state index
            pos_idx = i * env.grid_size + j
            status_value = 0
            for k, status in enumerate(slot_status):
                status_value += status * (2 ** k)
            state_idx = pos_idx * (2 ** len(env.parking_slots)) + status_value
            
            # Get optimal action
            optimal_action = np.argmax(q_table[state_idx])
            action_grid[i][j] = env.action_meaning[optimal_action]
            
            # Set grid color based on position type
            if (i, j) in env.parking_slots:
                slot_idx = env.parking_slots.index((i, j))
                grid[i][j] = 3 if slot_status[slot_idx] == 1 else 2
            else:
                grid[i][j] = 1
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    cmap = plt.cm.get_cmap('viridis', 4)
    im = plt.imshow(grid, cmap=cmap, vmin=0, vmax=3)
    
    # Add action symbols to each cell
    for i in range(env.grid_size):
        for j in range(env.grid_size):
            color = 'white' if (i, j) in env.parking_slots else 'black'
            plt.text(j, i, action_grid[i][j], 
                    ha="center", va="center", 
                    color=color, fontsize=20, fontweight='bold')
    
    # Add colorbar legend
    cbar = plt.colorbar(im, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(['Area Kosong', 'Area Jalan', 'Slot Kosong', 'Slot Terisi'])
    
    plt.title(f'Kebijakan Optimal yang Dipelajari\nStatus Slot: {slot_status}', fontsize=14)
    plt.xticks(np.arange(env.grid_size))
    plt.yticks(np.arange(env.grid_size))
    plt.grid(color='black', linestyle='-', linewidth=1)
    plt.tight_layout()
    
    # Save plot
    timestamp = int(time.time())
    plot_filename = f"policy_{timestamp}_{''.join(str(s) for s in slot_status)}.png"
    plot_path = os.path.join(STATIC_IMG_DIR, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    
    return plot_filename

def run_demo(env, q_table):
    """Run a demonstration of the optimal policy"""
    state = env.reset()
    total_reward = 0
    done = False
    step = 0
    max_steps = 30
    history = []
    
    while not done and step < max_steps:
        # Select optimal action
        action = np.argmax(q_table[state])
        
        # Take action
        new_state, reward, done = env.step(action)
        
        # Record state for visualization
        grid_state = get_grid_state(env)
        history.append({
            'step': step + 1,
            'action': env.action_meaning[action],
            'reward': reward,
            'grid': grid_state
        })
        
        # Update state and reward
        state = new_state
        total_reward += reward
        step += 1
        
        if done:
            break
    
    # Return final state for display
    final_grid = get_grid_state(env)
    
    return {
        'steps': step,
        'total_reward': total_reward,
        'grid': final_grid,
        'history': history
    }

def get_grid_state(env):
    """Get current grid state for visualization"""
    grid = [['.' for _ in range(env.grid_size)] for _ in range(env.grid_size)]
    
    # Mark parking slots
    for i, (r, c) in enumerate(env.parking_slots):
        if env.slot_status[i] == 0:
            grid[r][c] = 'O'  # Empty slot
        else:
            grid[r][c] = 'X'  # Occupied slot
    
    # Mark agent position
    r, c = env.agent_pos
    grid[r][c] = 'A'
    
    return grid

def run_hyperparameter_experiment():
    """Run experiment comparing different hyperparameters"""
    global env
    
    # Hyperparameter values to compare
    gamma_values = [0.1, 0.9]
    alpha = 0.9
    episodes = 500  # Reduced for demo purposes
    
    plt.figure(figsize=(12, 6))
    
    for gamma in gamma_values:
        # Train model with specific gamma
        env = ParkingEnvironment()  # Create new environment
        q_table, rewards_history = q_learning_train(
            env=env,
            episodes=episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=1.0,
            min_epsilon=0.01,
            decay_rate=0.005
        )
        
        # Calculate moving average
        rewards_series = pd.Series(rewards_history)
        rolling_mean = rewards_series.rolling(window=50, min_periods=1).mean()
        
        # Plot results
        plt.plot(rolling_mean, linewidth=2, label=f'gamma = {gamma}')
    
    # Finalize plot
    plt.title('Perbandingan Kurva Pembelajaran dengan Discount Factor Berbeda', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Rata-rata Reward (Moving Average)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    timestamp = int(time.time())
    plot_filename = f"hyperparam_comp_{timestamp}.png"
    plot_path = os.path.join(STATIC_IMG_DIR, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    
    return plot_filename

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true')