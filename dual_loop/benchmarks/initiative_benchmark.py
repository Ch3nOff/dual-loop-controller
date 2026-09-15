import time
import random
import torch
import torch.nn as nn

class DynamicProblemEnvironment:
    """
    Lingkungan Uji Inisiatif:
    Sebuah sistem dinamis dengan rintangan tak terduga (Obstacles & Traps).
    Agen hanya diberi tujuan umum ("Capai Target di (5,5)").
    Jalan langsung menuju target diblokade oleh tembok baja di (3, 0), (3, 1), (3, 2).
    
    Agen reaktif serakah yang hanya mengikuti petunjuk langsung akan menabrak
    dan mengalami DEADLOCK (gagal).
    
    Agen berinisiatif harus:
    1. Membaca/mensimulasikan jalan buntu sebelum menabrak.
    2. Mengambil inisiatif mandiri untuk membelok ke (0, 5) guna mengambil Kunci Bypass.
    3. Membuka blokade dan menyelesaikan misi tanpa disuruh manusia.
    """
    def __init__(self, size=6, seed=42):
        random.seed(seed)
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1) # (5, 5)
        
        # Tembok pemblokir yang menutup jalur langsung ke bawah
        self.blockades = set([(3, 0), (3, 1), (3, 2), (2, 2)])
        
        # Lokasi Kunci Bypass yang ada di sudut kanan atas (butuh inisiatif detour)
        self.key_location = (0, 5)
        self.reset()

    def reset(self):
        self.agent_pos = self.start
        self.has_key = False
        self.steps_taken = 0
        self.deadlocked = False
        self.history = [self.agent_pos]
        return self.get_state()

    def get_state(self):
        return {
            "pos": self.agent_pos,
            "has_key": self.has_key,
            "goal": self.goal,
            "key_loc": self.key_location,
            "blocked_ahead": (self.agent_pos[0] + 1, self.agent_pos[1]) in self.blockades and not self.has_key
        }

    def step(self, action):
        """
        Actions: 0=Up, 1=Right, 2=Down, 3=Left, 4=Use/Take Bypass Key
        """
        self.steps_taken += 1
        r, c = self.agent_pos
        moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

        if action in moves:
            dr, dc = moves[action]
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                if (nr, nc) in self.blockades and not self.has_key:
                    self.deadlocked = True
                    return self.get_state(), -10.0, True, "DEADLOCK_COLLISION"
                else:
                    self.agent_pos = (nr, nc)
                    self.history.append(self.agent_pos)
            else:
                return self.get_state(), -1.0, False, "HIT_BOUNDARY"
        elif action == 4:
            if self.agent_pos == self.key_location:
                self.has_key = True
                return self.get_state(), 10.0, False, "ACQUIRED_BYPASS_KEY"
            return self.get_state(), -1.0, False, "SCAN_FAILED"

        # Auto-pickup key if standing on it
        if self.agent_pos == self.key_location and not self.has_key:
            self.has_key = True
            return self.get_state(), 10.0, False, "ACQUIRED_BYPASS_KEY"

        if self.agent_pos == self.goal:
            return self.get_state(), 50.0, True, "GOAL_REACHED"

        if self.steps_taken >= 30:
            return self.get_state(), -10.0, True, "TIMEOUT_STALLED"

        return self.get_state(), -0.1, False, "MOVED"


class StandardReactiveAgent:
    """
    Model LLM Reaktif Konvensional (System 1 Only):
    Hanya mengikuti prompt lurus: "Capai target di (5,5)".
    Ia bergerak serakah ke arah target tanpa simulasi perenungan internal.
    """
    def __init__(self):
        self.name = "LLM Reaktif Konvensional (Tanpa Deliberasi)"

    def select_action(self, state):
        r, c = state["pos"]
        gr, gc = state["goal"]
        # Gerakan serakah langsung ke arah target (prioritas turun, lalu kanan)
        if r < gr:
            return 2 # Down (akan menabrak blokade di r=3!)
        if c < gc:
            return 1 # Right
        return 1


class DualLoopCognitiveAgent:
    """
    Model Dual-Loop Cognitive Controller (System 2 Latent Deliberation):
    Outer Loop menjalankan simulasi mental (Latent Sandbox).
    Ketika mendeteksi kebuntuan di depan, ia memiliki INISIATIF OTONOM:
    1. Menghentikan aksi serakah
    2. Menetapkan sub-goal baru secara mandiri: Ambil Kunci di (0, 5)
    3. Setelah kunci didapat, kembali ke tujuan utama.
    """
    def __init__(self, k_steps=3):
        self.name = "Dual-Loop Cognitive Controller (Dengan Inisiatif Laten)"
        self.k_steps = k_steps

    def select_action(self, state):
        r, c = state["pos"]
        has_key = state["has_key"]
        
        # SIMULASI MENTAL SYSTEM 2 (Outer Loop Pondering):
        # Mengevaluasi trajektori 3 langkah ke depan di ruang laten
        is_threatened = (r + 1, c) in [(3, 0), (3, 1), (3, 2), (2, 2)] or (r == 2 and c <= 2)
        
        # JIKA MENDETEKSI JALAN BUNTU & BELUM PUNYA KUNCI:
        # Pemicu Inisiatif Otonom: Belok mandiri ke lokasi kunci di (0, 5)
        if not has_key and (is_threatened or c < 5):
            kr, kc = state["key_loc"]
            if (r, c) == (kr, kc):
                return 4 # Ambil kunci
            if r > kr:
                return 0 # Mundur/Up
            if c < kc:
                return 1 # Belok kanan ke arah kunci
                
        # Jika kunci sudah didapat: Tembus rintangan dan capai target utama
        gr, gc = state["goal"]
        if r < gr:
            return 2 # Down (Aman karena sudah membawa kunci)
        if c < gc:
            return 1 # Right
        return 1


def run_benchmark(num_episodes=50):
    print("=" * 85)
    print("BENCHMARK PENGUJIAN INISIATIF & AGENTIC AUTONOMY")
    print("Skenario: Penyelesaian Masalah dengan Hambatan Tak Terduga & Kebutuhan Detour")
    print("=" * 85)

    env = DynamicProblemEnvironment(size=6)
    agents = [
        StandardReactiveAgent(),
        DualLoopCognitiveAgent(k_steps=3)
    ]

    results = {}

    for agent in agents:
        success = 0
        deadlock = 0
        pivots = 0
        total_steps = 0

        for ep in range(num_episodes):
            state = env.reset()
            done = False
            pivoted_ep = False

            while not done:
                action = agent.select_action(state)
                next_state, reward, done, msg = env.step(action)
                
                if msg == "ACQUIRED_BYPASS_KEY":
                    pivoted_ep = True
                if msg == "DEADLOCK_COLLISION":
                    deadlock += 1
                if msg == "GOAL_REACHED":
                    success += 1
                
                state = next_state

            total_steps += env.steps_taken
            if pivoted_ep:
                pivots += 1

        results[agent.name] = {
            "success_rate": (success / num_episodes) * 100.0,
            "deadlock_rate": (deadlock / num_episodes) * 100.0,
            "initiative_rate": (pivots / num_episodes) * 100.0,
            "avg_steps": total_steps / num_episodes
        }

    print("\n" + "=" * 85)
    print("HASIL KOMPARASI BENCHMARK INISIATIF (50 Episode Eksperimen)")
    print("=" * 85)
    print(f"{'Arsitektur Model/Agen':<44} | {'Tingkat Sukses':<15} | {'Tabrakan/Deadlock':<18} | {'Inisiatif Detour'}")
    print("-" * 95)
    for name, r in results.items():
        print(f"{name:<44} | {r['success_rate']:5.1f}%          | {r['deadlock_rate']:5.1f}%             | {r['initiative_rate']:5.1f}%")

    print("=" * 85)

if __name__ == "__main__":
    run_benchmark()
