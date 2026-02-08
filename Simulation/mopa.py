import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from scipy.interpolate import interp1d

def read_signal(signal_path):
    timestamps, signal = [], []
    with open(signal_path) as file:
        file.readline()
        for line in file:
            splitted_line = line.strip().split(',')
            if len(splitted_line) >= 2:
                timestamps.append(float(splitted_line[0]))
                signal.append(float(splitted_line[1]))
    timestamps = np.array(timestamps)
    signal = np.array(signal)
    fs = 1.0 / np.median(np.diff(timestamps))
    print(f"Red Signal: length: {timestamps[-1]:.2f}, sample rate {fs:.0f}")
    return timestamps, signal, fs

def compute_stft(signal, fs, nperseg=512, noverlap=None):
    if noverlap is None:
        noverlap = nperseg // 2
    f, t, Zxx = stft(signal, fs, 'hann', nperseg, noverlap)
    Sxx = np.abs(Zxx)
    return  f, t, Sxx

def normalize_spectrum(Sxx):
    rms = np.sqrt(np.mean(Sxx**2))
    return Sxx / rms if rms > 0 else Sxx

def compute_pdf(Axx, f, orders, min_omega, max_omega, nomega=1000, freq_bias_strength=0.75):
    """
    Compute PDF with optional linear frequency bias.
    
    freq_bias_strength: Controls preference for lower frequencies
        0.0 = no bias
        1.0 = full linear bias (weight goes from 1 at 0 Hz to 0 at f_max)
        
    The weight applied to each magnitude is: weight = 1 - strength * (freq / f_max)
    """
    f_max = f[-1]
    omega = np.linspace(min_omega, max_omega, nomega)
    interpolated_spectrum = interp1d(f, Axx)

    pdf = np.ones_like(omega)
    for order in orders:
        for i, w in enumerate(omega):
            freq = order * w
            if freq < f_max:
                magnitude = max(interpolated_spectrum(freq), 1e-10)
                
                # Apply linear frequency bias to prefer lower frequencies
                if freq_bias_strength > 0:
                    weight = 1.0 - freq_bias_strength * (freq / f_max)
                    weight = max(weight, 0.01)
                    magnitude *= weight
                
                pdf[i] *= magnitude

    total = np.trapezoid(pdf, omega)
    normalized_pdf = np.zeros_like(pdf)
    if total > 0:
        normalized_pdf = pdf / total

    return normalized_pdf, omega

def compute_pdf_map(Sxx, f, orders, min_omega, max_omega, freq_bias_strength=0.75):
    _, n_time = Sxx.shape
    
    pdf_map = None
    for i in range(n_time):
        Axx = Sxx[:,i]
        pdf, omega = compute_pdf(Axx, f, orders, min_omega, max_omega, freq_bias_strength=freq_bias_strength)

        if pdf_map is None:
            pdf_map = np.zeros((len(omega), n_time))

        pdf_map[:,i] = pdf

    return pdf_map, omega

def compute_ias(pdf_map, omega, sigma=0.5, weight=30.0, Sxx=None, amplitude_threshold=None, wait_steps=2):
    _, n_time = pdf_map.shape
    ias = np.zeros(n_time)
    
    # First compute raw IAS estimates
    raw_ias = np.zeros(n_time)
    raw_ias[0] = omega[np.argmax(pdf_map[:,0])]

    for i in range(1, n_time):
        gaussian_bell = np.exp(-(omega - raw_ias[i-1])**2 / (2*sigma**2))
        biased_pdf = pdf_map[:,i-1] * (1 + gaussian_bell * weight)
        raw_ias[i] = omega[np.argmax(biased_pdf)]

    # Compute zero mask based on amplitude threshold
    if Sxx is not None and amplitude_threshold is not None:
        max_amplitude = np.max(Sxx, axis=0)
        should_be_zero = max_amplitude < amplitude_threshold
    else:
        should_be_zero = np.zeros(n_time, dtype=bool)
    
    # Apply hysteresis: if previous was zero, wait for consecutive non-zero steps
    consecutive_nonzero = 0
    was_zero = True
    
    for i in range(n_time):
        if should_be_zero[i]:
            ias[i] = 0.0
            consecutive_nonzero = 0
            was_zero = True
        else:
            if was_zero:
                consecutive_nonzero += 1
                if consecutive_nonzero >= wait_steps:
                    ias[i] = raw_ias[i]
                    was_zero = False
                else:
                    ias[i] = 0.0
            else:
                ias[i] = raw_ias[i]
    
    return ias

if __name__ == "__main__":
    timestamps, signal, fs = read_signal("../../mopa_input/rampenfunktion_beschleunigung_2khz.csv")
    # timestamps, signal, fs = read_signal("../../mopa_input/treppenfunktion_beschleunigung_2khz.csv")
    # timestamps, signal, fs = read_signal("../../mopa_input/konstant.csv")

    plt.figure()
    plt.plot(timestamps, signal)
    plt.show()

    f, t, Sxx = compute_stft(signal, fs, nperseg=4 * 512)
    Sxx_normalized = normalize_spectrum(Sxx)
    Sxx_normalized_dB = 20 * np.log10(Sxx_normalized)

    plt.figure()
    plt.pcolormesh(t, f, Sxx_normalized_dB)
    plt.show()

    orders = [1, 4]
    min_omega, max_omega = 5, 24 

    pdf_map, omega = compute_pdf_map(Sxx_normalized, f, orders, min_omega, max_omega, freq_bias_strength=0.75)

    plt.figure()
    plt.pcolormesh(t, omega, pdf_map)
    plt.show()

    ias = compute_ias(pdf_map, omega, Sxx=Sxx_normalized, amplitude_threshold=1.0, wait_steps=2)

    # Save IAS speed estimation to CSV file
    output_path = "ias_speed_estimation_rampenfunktion.csv"
    np.savetxt(output_path, np.column_stack((t, ias)), delimiter=',', header='time,ias', comments='')
    print(f"IAS speed estimation saved to {output_path}")

    plt.figure()
    plt.plot(t, ias)
    plt.show()