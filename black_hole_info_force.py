import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

def load_strain_data(file_path):
    with h5py.File(file_path, 'r') as f:
        strain = f['strain']['Strain'][:]
        dt = f['strain']['Strain'].attrs['Xspacing']
    return strain, dt

def extract_time_segment(strain, dt, start_time, end_time):
    start_idx = int(start_time / dt)
    end_idx = int(end_time / dt)
    return strain[start_idx:end_idx], np.arange(start_idx, end_idx) * dt

def compute_energy_dissipation(strain_segment, dt):
    # Energy dissipation proxy: integral of strain squared over time
    return np.sum(strain_segment**2) * dt

def calculate_hawking_temperature(h, c, G, M, k_B):
    return (h * c**3) / (8 * np.pi * G * M * k_B)

def calculate_entropy_flux(energy_dissipation, temperature):
    # dE = T dS -> dS/dt = Energy rate / T
    return energy_dissipation / temperature

def calculate_information_force(h, c, v, phi_s, G, M, k_B):
    numerator = h * c * v * phi_s
    denominator = 8 * np.pi * G * M * k_B
    return numerator / denominator

def main():
    # File and segment settings
    file_path = "H-H1_GWOSC_16KHZ_R1-1126259447-32.hdf5"
    start_time = 0.45  # seconds
    end_time = 0.55    # seconds
    
    # Physical constants
    h = 6.62607015e-34      # Planck constant (J·s)
    c = 3e8                 # Speed of light (m/s)
    G = 6.67430e-11         # Gravitational constant (m^3·kg^-1·s^-2)
    k_B = 1.380649e-23      # Boltzmann constant (J/K)
    
    # Parameters (example black hole mass and info velocity)
    M = 1e30                # Black hole mass in kg (adjust accordingly)
    v = 1e8                 # Entropy carrier velocity (m/s), assumed
    
    # Load strain data and time step
    strain_data, dt = load_strain_data(file_path)
    
    # Extract segment around merger
    strain_segment, time_segment = extract_time_segment(strain_data, dt, start_time, end_time)
    
    # Basic data sanity check
    if len(strain_segment) == 0:
        raise ValueError("Extracted strain segment is empty. Check start/end times.")
    
    # Energy dissipation estimate
    energy_dissipation = compute_energy_dissipation(strain_segment, dt)
    
    # Hawking temperature for black hole mass M
    T_hawking = calculate_hawking_temperature(h, c, G, M, k_B)
    
    # Entropy flux (J/K/s)
    phi_s = calculate_entropy_flux(energy_dissipation, T_hawking)
    
    # Information Force (Newtons)
    F_info = calculate_information_force(h, c, v, phi_s, G, M, k_B)
    
    # Pressure (P = F / v)
    P_info = F_info / v
    
    # Energy flow rate (W)
    energy_flow_rate = T_hawking * phi_s
    
    # Output results
    print(f"Energy Dissipation (J): {energy_dissipation:.4e}")
    print(f"Hawking Temperature (K): {T_hawking:.4e}")
    print(f"Entropy Flux (J/K/s): {phi_s:.4e}")
    print(f"Information Force (N): {F_info:.4e}")
    print(f"Pressure (Pa): {P_info:.4e}")
    print(f"Energy Flow Rate (W): {energy_flow_rate:.4e}")
    
    # Optional: plot strain segment and amplitude spectrum
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(time_segment, strain_segment)
    plt.title("Strain vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    
    # FFT of the strain segment
    N = len(strain_segment)
    yf = rfft(strain_segment)
    xf = rfftfreq(N, dt)
    amplitude_spectrum = np.abs(yf)
    
    plt.subplot(1, 2, 2)
    plt.plot(xf, amplitude_spectrum)
    plt.title("Amplitude Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
