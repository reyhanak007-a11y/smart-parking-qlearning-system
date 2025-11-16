import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from collections import defaultdict
import math
import time

class ParkingEnvironment:
    """
    Kelas yang merepresentasikan lingkungan parkir sederhana.
    - Grid 3x3 dengan 4 slot parkir di posisi (0,0), (0,2), (2,0), dan (2,2)
    - Setiap slot parkir bisa dalam status kosong (0) atau terisi (1)
    - Agen harus menemukan slot parkir kosong dan parkir di sana
    """
    def __init__(self):

        self.grid_size = 3
        self.num_rows = self.grid_size
        self.num_cols = self.grid_size
        

        self.parking_slots = [(0, 0), (0, 2), (2, 0), (2, 2)]
        

        self.action_space = [0, 1, 2, 3, 4]
        self.action_meaning = {
            0: '←',  # Kiri
            1: '→',  # Kanan
            2: '↑',  # Atas
            3: '↓',  # Bawah
            4: 'P'   # Parkir
        }
        

        self.reset()
        
    def reset(self):
        """Reset environment ke kondisi awal untuk episode baru"""

        self.agent_pos = (1, 1)
        

        self.slot_status = [random.choice([0, 1]) for _ in range(len(self.parking_slots))]
        

        self.done = False
        self.parked = False
        
        return self._get_state()
    
    def _get_state(self):
        """
        Menggabungkan posisi agen dan status slot parkir menjadi satu state integer
        - Posisi agen: (x,y) di grid 3x3
        - Status slot parkir: 4 slot dengan status 0/1
        - Total state space: 9 posisi x 2^4 kombinasi status = 144 state
        """
        pos_idx = self.agent_pos[0] * self.grid_size + self.agent_pos[1]
        

        status_value = 0
        for i, status in enumerate(self.slot_status):
            status_value += status * (2 ** i)
        

        state = pos_idx * (2 ** len(self.parking_slots)) + status_value
        return state
    
    def step(self, action):
        """
        Melakukan action dan mengembalikan new_state, reward, done
        Args:
            action: integer (0-4) yang merepresentasikan aksi yang dipilih
        
        Returns:
            new_state: state baru setelah aksi
            reward: reward yang diterima
            done: boolean apakah episode selesai
        """
        reward = -1  # Penalty default per langkah
        
        if self.done:
            return self._get_state(), reward, True
        

        if action == 0:  # Kiri
            new_pos = (self.agent_pos[0], max(0, self.agent_pos[1] - 1))

            if new_pos[1] == self.agent_pos[1] and new_pos[1] == 0:
                reward = -3
            self.agent_pos = new_pos
            
        elif action == 1:  # Kanan
            new_pos = (self.agent_pos[0], min(self.grid_size - 1, self.agent_pos[1] + 1))

            if new_pos[1] == self.agent_pos[1] and new_pos[1] == self.grid_size - 1:
                reward = -3
            self.agent_pos = new_pos
            
        elif action == 2:  # Atas
            new_pos = (max(0, self.agent_pos[0] - 1), self.agent_pos[1])

            if new_pos[0] == self.agent_pos[0] and new_pos[0] == 0:
                reward = -3
            self.agent_pos = new_pos
            
        elif action == 3:  # Bawah
            new_pos = (min(self.grid_size - 1, self.agent_pos[0] + 1), self.agent_pos[1])

            if new_pos[0] == self.agent_pos[0] and new_pos[0] == self.grid_size - 1:
                reward = -3
            self.agent_pos = new_pos
            
        elif action == 4:  # Parkir

            current_pos = self.agent_pos
            if current_pos in self.parking_slots:
                slot_idx = self.parking_slots.index(current_pos)

                if self.slot_status[slot_idx] == 0:

                    reward = 10
                    self.slot_status[slot_idx] = 1  # Tandai slot sebagai terisi
                    self.parked = True
                    self.done = True
                else:

                    reward = -5
            else:

                reward = -5
        

        if all(status == 1 for status in self.slot_status):
            self.done = True
        
        new_state = self._get_state()
        return new_state, reward, self.done
    
    def render(self):
        """Menampilkan visualisasi grid parkir saat ini"""
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        

        for i, (r, c) in enumerate(self.parking_slots):
            if self.slot_status[i] == 0:
                grid[r][c] = 'O'  # Slot kosong
            else:
                grid[r][c] = 'X'  # Slot terisi
        

        r, c = self.agent_pos
        grid[r][c] = 'A'
        

        print("\nArea Parkir:")
        for row in grid:
            print(' '.join(row))
        print(f"Status Slot: {self.slot_status}")
        print(f"Parked: {self.parked}, Done: {self.done}")

def q_learning_train(env, episodes=1000, alpha=0.9, gamma=0.9, epsilon=1.0, min_epsilon=0.01, decay_rate=0.005):
    """
    Melatih agen Q-Learning pada environment parkir
    
    Args:
        env: instance dari ParkingEnvironment
        episodes: jumlah episode pelatihan
        alpha: learning rate
        gamma: discount factor
        epsilon: nilai epsilon awal untuk strategi epsilon-greedy
        min_epsilon: nilai minimum epsilon
        decay_rate: tingkat penurunan epsilon
    
    Returns:
        q_table: Q-Table yang sudah dilatih
        rewards_history: riwayat reward per episode
    """

    state_size = env.grid_size * env.grid_size * (2 ** len(env.parking_slots))
    action_size = len(env.action_space)
    q_table = np.zeros((state_size, action_size))
    
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        

        max_steps = 30
        step = 0
        
        while not done and step < max_steps:

            if random.uniform(0, 1) < epsilon:
                action = random.choice(env.action_space)  # Eksplorasi
            else:
                action = np.argmax(q_table[state])  # Eksploitasi
            

            new_state, reward, done = env.step(action)
            

            old_value = q_table[state, action]
            next_max = np.max(q_table[new_state])
            

            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[state, action] = new_value
            

            state = new_state
            total_reward += reward
            step += 1
        

        rewards_history.append(total_reward)
        

        epsilon = max(min_epsilon, epsilon * math.exp(-decay_rate * episode))
        

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode+1}/{episodes}, Avg Reward (last 100): {avg_reward:.2f}, Epsilon: {epsilon:.4f}")
    
    return q_table, rewards_history

def plot_learning_curve(rewards_history, window_size=100, title="Kurva Pembelajaran Q-Learning"):
    """
    Membuat visualisasi kurva pembelajaran dengan moving average
    
    Args:
        rewards_history: list riwayat reward per episode
        window_size: ukuran window untuk moving average
        title: judul plot
    """
    rewards_series = pd.Series(rewards_history)
    rolling_mean = rewards_series.rolling(window=window_size, min_periods=1).mean()
    
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_history, alpha=0.3, color='blue', label='Reward per Episode')
    plt.plot(rolling_mean, color='green', linewidth=2, label=f'Moving Average ({window_size} episode)')
    plt.title(title, fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('learning_curve.png')
    plt.show()

def visualize_policy(env, q_table, title="Kebijakan Optimal pada Grid Parkir"):
    """
    Memvisualisasikan kebijakan optimal yang dipelajari dari Q-Table
    
    Args:
        env: instance dari ParkingEnvironment
        q_table: Q-Table yang sudah dilatih
        title: judul visualisasi
    """

    state = env.reset()
    

    slot_status = env.slot_status.copy()
    

    grid = np.zeros((env.grid_size, env.grid_size), dtype=int)
    action_grid = [['' for _ in range(env.grid_size)] for _ in range(env.grid_size)]
    

    for i in range(env.grid_size):
        for j in range(env.grid_size):

            pos_idx = i * env.grid_size + j
            status_value = 0
            for k, status in enumerate(slot_status):
                status_value += status * (2 ** k)
            state_idx = pos_idx * (2 ** len(env.parking_slots)) + status_value
            

            optimal_action = np.argmax(q_table[state_idx])
            action_grid[i][j] = env.action_meaning[optimal_action]
            

            if (i, j) in env.parking_slots:
                slot_idx = env.parking_slots.index((i, j))
                grid[i][j] = 3 if slot_status[slot_idx] == 1 else 2  # 3=terisi, 2=kosong
            else:
                grid[i][j] = 1  # Area jalan
    

    plt.figure(figsize=(10, 8))
    cmap = plt.cm.get_cmap('viridis', 4)
    

    im = plt.imshow(grid, cmap=cmap, vmin=0, vmax=3)
    

    for i in range(env.grid_size):
        for j in range(env.grid_size):
            color = 'white' if (i, j) in env.parking_slots else 'black'
            plt.text(j, i, action_grid[i][j], 
                    ha="center", va="center", 
                    color=color, fontsize=20, fontweight='bold')
    

    cbar = plt.colorbar(im, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(['Area Kosong', 'Area Jalan', 'Slot Kosong', 'Slot Terisi'])
    
    plt.title(f'{title}\nStatus Slot: {slot_status}', fontsize=14)
    plt.xticks(np.arange(env.grid_size))
    plt.yticks(np.arange(env.grid_size))
    plt.grid(color='black', linestyle='-', linewidth=1)
    plt.tight_layout()
    

    filename = f"policy_visualization_{''.join(str(s) for s in slot_status)}.png"
    plt.savefig(filename)
    plt.show()
    
    return filename

def run_hyperparameter_experiment():
    """
    Menjalankan eksperimen untuk membandingkan dampak hyperparameter berbeda
    Khususnya discount factor (gamma)
    """
    print("\n" + "="*50)
    print("MEMULAI EKSPERIMEN HIPERPARAMETER")
    print("="*50)
    

    env = ParkingEnvironment()
    

    gamma_values = [0.1, 0.9]  # Discount factor yang berbeda
    alpha = 0.9
    episodes = 1000
    
    results = {}
    
    for gamma in gamma_values:
        print(f"\nMelatih model dengan gamma = {gamma}")
        start_time = time.time()
        

        q_table, rewards_history = q_learning_train(
            env=env,
            episodes=episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=1.0,
            min_epsilon=0.01,
            decay_rate=0.005
        )
        
        training_time = time.time() - start_time
        print(f"Waktu pelatihan: {training_time:.2f} detik")
        

        results[gamma] = {
            'rewards_history': rewards_history,
            'q_table': q_table
        }
        

        print(f"Memvisualisasikan kebijakan untuk gamma = {gamma}")
        visualize_policy(env, q_table, title=f"Kebijakan Optimal (gamma={gamma})")
    

    plt.figure(figsize=(12, 6))
    
    window_size = 100
    for gamma, data in results.items():
        rewards_series = pd.Series(data['rewards_history'])
        rolling_mean = rewards_series.rolling(window=window_size, min_periods=1).mean()
        plt.plot(rolling_mean, linewidth=2, label=f'gamma = {gamma}')
    
    plt.title('Perbandingan Kurva Pembelajaran dengan Discount Factor Berbeda', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Rata-rata Reward (Moving Average)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('hyperparameter_comparison.png')
    plt.show()
    
    return results
def save_model(q_table, filename="q_table_model.npy"):
    """
    Menyimpan Q-Table hasil training ke dalam file
    
    Args:
        q_table: Q-Table yang telah dilatih (numpy array)
        filename: Nama file untuk menyimpan model
    
    Returns:
        None
    """

    np.save(filename, q_table)
    

    metadata = {
        'state_size': q_table.shape[0],
        'action_size': q_table.shape[1],
        'training_date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'algorithm': 'Q-Learning'
    }
    
    metadata_filename = filename.replace('.npy', '_metadata.json')
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Model berhasil disimpan ke {filename}")
    print(f"Metadata model disimpan ke {metadata_filename}")


def load_model(filename="q_table_model.npy"):
    """
    Memuat Q-Table dari file
    
    Args:
        filename: Nama file yang berisi model
    
    Returns:
        q_table: Q-Table yang telah dimuat (numpy array)
    """
    try:

        if not os.path.exists(filename):
            raise FileNotFoundError(f"File model {filename} tidak ditemukan")
        

        q_table = np.load(filename)
        

        metadata_filename = filename.replace('.npy', '_metadata.json')
        if os.path.exists(metadata_filename):
            with open(metadata_filename, 'r') as f:
                metadata = json.load(f)
            print(f"Model dimuat dengan metadata: {metadata}")
        else:
            print("Model dimuat tanpa metadata")
        
        print(f"Q-Table berhasil dimuat dari {filename}")
        print(f"Bentuk Q-Table: {q_table.shape}")
        return q_table
    
    except Exception as e:
        print(f"Error saat memuat model: {e}")
        return None


def save_training_history(rewards_history, filename="training_history.npy"):
    """
    Menyimpan riwayat training (reward per episode) untuk analisis lebih lanjut
    
    Args:
        rewards_history: List berisi total reward per episode
        filename: Nama file untuk menyimpan riwayat training
    
    Returns:
        None
    """
    np.save(filename, rewards_history)
    print(f"Riwayat training berhasil disimpan ke {filename}")


def load_training_history(filename="training_history.npy"):
    """
    Memuat riwayat training dari file
    
    Args:
        filename: Nama file yang berisi riwayat training
    
    Returns:
        rewards_history: List berisi total reward per episode
    """
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File riwayat training {filename} tidak ditemukan")
        
        rewards_history = np.load(filename)
        print(f"Riwayat training berhasil dimuat dari {filename}")
        return rewards_history.tolist()
    
    except Exception as e:
        print(f"Error saat memuat riwayat training: {e}")
        return None
    
def demo_optimal_policy(env, q_table):
    """
    Mendemonstrasikan kebijakan optimal yang dipelajari
    """
    print("\n" + "="*50)
    print("DEMONSTRASI KEBIJAKAN OPTIMAL")
    print("="*50)
    
    state = env.reset()
    total_reward = 0
    done = False
    step = 0
    
    print("Status awal environment:")
    env.render()
    
    max_steps = 30
    
    while not done and step < max_steps:

        action = np.argmax(q_table[state])
        

        new_state, reward, done = env.step(action)
        

        action_name = "Parkir" if action == 4 else \
                     ["Kiri", "Kanan", "Atas", "Bawah"][action]
        print(f"\nLangkah {step+1}:")
        print(f"Action: {action_name} ({env.action_meaning[action]})")
        print(f"Reward: {reward}")
        env.render()
        

        state = new_state
        total_reward += reward
        step += 1
        
        if done:
            break
    
    print(f"\nTotal Reward: {total_reward}")
    print(f"Episode selesai dalam {step} langkah")
    
    return total_reward, step

if __name__ == "__main__":

    env = ParkingEnvironment()
    
    print("="*60)
    print("IMPLEMENTASI SISTEM PARKIR OTOMATIS DENGAN Q-LEARNING")
    print("="*60)
    
    model_file = "q_table.npy"
    if os.path.exists(model_file):
        print(f"\nModel ditemukan di {model_file}, memuat model yang sudah ada...")
        q_table = load_model(model_file)
        rewards_history = load_training_history(model_file.replace('.npy', '_history.npy'))
    else:
        print("\nModel tidak ditemukan, melatih model baru...")
        
        print("\nContoh Environment Awal:")
        env.render()
        
        print("\nMemulai pelatihan Q-Learning...")
        print("Hyperparameter:")
        print("- Learning Rate (alpha): 0.9")
        print("- Discount Factor (gamma): 0.9")
        print("- Total Episode: 1000")
        print("- Max Steps per Episode: 30")
        
        q_table, rewards_history = q_learning_train(
            env=env,
            episodes=1000,
            alpha=0.9,
            gamma=0.9,
            epsilon=1.0,
            min_epsilon=0.01,
            decay_rate=0.005
        )
        
        save_model(q_table, model_file)
        save_training_history(rewards_history, model_file.replace('.npy', '_history.npy'))