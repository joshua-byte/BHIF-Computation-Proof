import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# === Constants ===
h = 6.62607015e-34      # Planck constant (J·s)
c = 3e8                 # Speed of light (m/s)
G = 6.67430e-11         # Gravitational constant (m^3·kg^-1·s^-2)
k_B = 1.380649e-23      # Boltzmann constant (J/K)
M = 1e30                # Black hole mass (kg)
v = 1e8                 # Information/entropy carrier velocity (m/s)

# === Load strain data ===
def load_strain_data(file_path):
    with h5py.File(file_path, 'r') as f:
        strain = f['strain']['Strain'][:]
        dt = f['strain']['Strain'].attrs['Xspacing']
    return strain, dt

# === Energy dissipation ===
def compute_energy_dissipation_local(strain_window, dt):
    return np.sum(strain_window**2) * dt

# === Hawking temperature ===
def calculate_hawking_temperature(h, c, G, M, k_B):
    return (h * c**3) / (8 * np.pi * G * M * k_B)

# === Entropy flux ===
def calculate_entropy_flux(energy_dissipation, temperature):
    return energy_dissipation / temperature

# === Information force ===
def calculate_information_force(h, c, v, phi_s, G, M, k_B):
    numerator = h * c * v * phi_s
    denominator = 8 * np.pi * G * M * k_B
    return numerator / denominator

def main():
    # === File and settings ===
    file_path = "H-H1_GWOSC_16KHZ_R1-1126259447-32.hdf5"
    start_time = 0.4
    end_time = 0.6
    window_size = 0.01     # 10 ms window
    step_size = 0.002      # 2 ms step

    # === Load data ===
    strain, dt = load_strain_data(file_path)
    time_array = np.arange(0, len(strain)) * dt

    # === Extract strain segment ===
    start_idx = int(start_time / dt)
    end_idx = int(end_time / dt)
    strain_segment = strain[start_idx:end_idx]
    time_segment = time_array[start_idx:end_idx]

    # === Plot 1: Strain vs Time ===
    plt.figure(figsize=(10, 4))
    plt.plot(time_segment, strain_segment, color='black')
    plt.title("Gravitational Wave Strain vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Plot 2: Amplitude Spectrum ===
    N = len(strain_segment)
    yf = rfft(strain_segment)
    xf = rfftfreq(N, dt)
    amplitude_spectrum = np.abs(yf)

    plt.figure(figsize=(10, 4))
    plt.plot(xf, amplitude_spectrum, color='purple')
    plt.title("Amplitude Spectrum of Strain Signal")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(0, 1000)  # Limit to relevant frequency band
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Sliding window for Entropy Flux and Info Force ===
    window_len = int(window_size / dt)
    step_len = int(step_size / dt)

    times = []
    phi_s_values = []
    F_info_values = []

    T_hawking = calculate_hawking_temperature(h, c, G, M, k_B)

    for i in range(0, len(strain_segment) - window_len, step_len):
        window = strain_segment[i:i + window_len]
        local_time = time_segment[i + window_len // 2]

        dE = compute_energy_dissipation_local(window, dt)
        phi_s = calculate_entropy_flux(dE, T_hawking)
        F_info = calculate_information_force(h, c, v, phi_s, G, M, k_B)

        times.append(local_time)
        phi_s_values.append(phi_s)
        F_info_values.append(F_info)

    # === Plot 3 & 4: Entropy Flux and Info Force ===
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(times, phi_s_values, color='darkorange')
    plt.title("Entropy Flux vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Entropy Flux (J/K/s)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(times, F_info_values, color='royalblue')
    plt.title("Information Force vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Information Force (N)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # === Print key outputs from full segment ===
    total_energy = compute_energy_dissipation_local(strain_segment, dt)
    phi_s_total = calculate_entropy_flux(total_energy, T_hawking)
    F_info_total = calculate_information_force(h, c, v, phi_s_total, G, M, k_B)
    P_info = F_info_total / v
    energy_flow_rate = T_hawking * phi_s_total

    print("\n=== Summary of Results ===")
    print(f"Energy Dissipation (J): {total_energy:.4e}")
    print(f"Hawking Temperature (K): {T_hawking:.4e}")
    print(f"Entropy Flux (J/K/s): {phi_s_total:.4e}")
    print(f"Information Force (N): {F_info_total:.4e}")
    print(f"Pressure (Pa): {P_info:.4e}")
    print(f"Energy Flow Rate (W): {energy_flow_rate:.4e}")

if __name__ == "__main__":
    main()
