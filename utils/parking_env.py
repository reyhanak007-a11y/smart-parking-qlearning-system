import random
import math

class ParkingEnvironment:
    """
    Kelas yang merepresentasikan lingkungan parkir sederhana.
    - Grid 3x3 dengan 4 slot parkir di posisi (0,0), (0,2), (2,0), dan (2,2)
    - Setiap slot parkir bisa dalam status kosong (0) atau terisi (1)
    - Agen harus menemukan slot parkir kosong dan parkir di sana
    """
    def __init__(self):
        # Dimensi grid parkir (3x3)
        self.grid_size = 3
        self.num_rows = self.grid_size
        self.num_cols = self.grid_size
        
        # Posisi slot parkir tetap
        self.parking_slots = [(0, 0), (0, 2), (2, 0), (2, 2)]
        
        # Action space: 0=Kiri, 1=Kanan, 2=Atas, 3=Bawah, 4=Parkir
        self.action_space = [0, 1, 2, 3, 4]
        self.action_meaning = {
            0: '←',  # Kiri
            1: '→',  # Kanan
            2: '↑',  # Atas
            3: '↓',  # Bawah
            4: 'P'   # Parkir
        }
        
        # Reset environment ke state awal
        self.reset()
        
    def reset(self):
        """Reset environment ke kondisi awal untuk episode baru"""
        # Posisi awal kendaraan di tengah grid (1,1)
        self.agent_pos = (1, 1)
        
        # Status slot parkir: 0=kosong, 1=terisi (diatur secara acak)
        self.slot_status = [random.choice([0, 1]) for _ in range(len(self.parking_slots))]
        
        # Episode selesai atau belum
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
        
        # Konversi status slot menjadi bilangan desimal
        status_value = 0
        for i, status in enumerate(self.slot_status):
            status_value += status * (2 ** i)
        
        # Kombinasi posisi dan status
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
        
        # Eksekusi action
        if action == 0:  # Kiri
            new_pos = (self.agent_pos[0], max(0, self.agent_pos[1] - 1))
            # Penalty jika tabrak dinding
            if new_pos[1] == self.agent_pos[1] and new_pos[1] == 0:
                reward = -3
            self.agent_pos = new_pos
            
        elif action == 1:  # Kanan
            new_pos = (self.agent_pos[0], min(self.grid_size - 1, self.agent_pos[1] + 1))
            # Penalty jika tabrak dinding
            if new_pos[1] == self.agent_pos[1] and new_pos[1] == self.grid_size - 1:
                reward = -3
            self.agent_pos = new_pos
            
        elif action == 2:  # Atas
            new_pos = (max(0, self.agent_pos[0] - 1), self.agent_pos[1])
            # Penalty jika tabrak dinding
            if new_pos[0] == self.agent_pos[0] and new_pos[0] == 0:
                reward = -3
            self.agent_pos = new_pos
            
        elif action == 3:  # Bawah
            new_pos = (min(self.grid_size - 1, self.agent_pos[0] + 1), self.agent_pos[1])
            # Penalty jika tabrak dinding
            if new_pos[0] == self.agent_pos[0] and new_pos[0] == self.grid_size - 1:
                reward = -3
            self.agent_pos = new_pos
            
        elif action == 4:  # Parkir
            # Cek apakah agen berada di slot parkir
            current_pos = self.agent_pos
            if current_pos in self.parking_slots:
                slot_idx = self.parking_slots.index(current_pos)
                # Cek apakah slot kosong
                if self.slot_status[slot_idx] == 0:
                    # Parkir berhasil
                    reward = 10
                    self.slot_status[slot_idx] = 1  # Tandai slot sebagai terisi
                    self.parked = True
                    self.done = True
                else:
                    # Coba parkir di slot yang sudah terisi
                    reward = -5
            else:
                # Coba parkir di luar area parkir
                reward = -5
        
        # Cek apakah semua slot sudah terisi
        if all(status == 1 for status in self.slot_status):
            self.done = True
        
        new_state = self._get_state()
        return new_state, reward, self.done
    
    def render(self):
        """Menampilkan visualisasi grid parkir saat ini (untuk debugging)"""
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Tandai slot parkir
        for i, (r, c) in enumerate(self.parking_slots):
            if self.slot_status[i] == 0:
                grid[r][c] = 'O'  # Slot kosong
            else:
                grid[r][c] = 'X'  # Slot terisi
        
        # Tandai posisi agen
        r, c = self.agent_pos
        grid[r][c] = 'A'
        
        # Cetak grid
        print("\nArea Parkir:")
        for row in grid:
            print(' '.join(row))
        print(f"Status Slot: {self.slot_status}")
        print(f"Parked: {self.parked}, Done: {self.done}")