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

def compute_pdf(Axx, f, orders, min_omega, max_omega, nomega=1000):
    f_max = f[-1]
    omega = np.linspace(min_omega, max_omega, nomega)
    interpolated_spectrum = interp1d(f, Axx)

    pdf = np.ones_like(omega)
    for order in orders:
        for i, w in enumerate(omega):
            freq = order * w
            if freq < f_max:
                magnitude = max(interpolated_spectrum(freq), 1e-10)
                pdf[i] *= magnitude

    total = np.trapezoid(pdf, omega)
    normalized_pdf = np.zeros_like(pdf)
    if total > 0:
        normalized_pdf = pdf / total

    return normalized_pdf, omega

def compute_pdf_map(Sxx, f, orders, min_omega, max_omega):
    _, n_time = Sxx.shape

    pdf_map = None
    for i in range(n_time):
        Axx = Sxx[:,i]
        pdf, omega = compute_pdf(Axx, f, orders, min_omega, max_omega)

        if pdf_map is None:
            pdf_map = np.zeros((len(omega), n_time))

        pdf_map[:,i] = pdf

    return pdf_map, omega

def rms_to_omega(rms, min_omega, max_omega, rms_max=2.0):
    return min_omega + (max_omega - min_omega) * np.clip(rms / rms_max, 0, 1)

def compute_ias(pdf_map, omega, Sxx, min_omega=5, max_omega=24,
                t_sigma=0.5, t_weight=40.0,
                r_sigma=1.0, r_weight=15.0, disagree_threshold=3.0,
                rms_threshold=0.18, wait_steps=3):
    _, n_time = pdf_map.shape
    ias = np.zeros(n_time)

    rms = np.sqrt(np.mean(Sxx**2, axis=0))
    omega_from_rms = rms_to_omega(rms, min_omega, max_omega)

    raw_ias = np.zeros(n_time)
    raw_ias[0] = omega[np.argmax(pdf_map[:,0])]

    for i in range(1, n_time):
        tracking_g = np.exp(-(omega - raw_ias[i-1])**2 / (2*t_sigma**2))

        disagreement = abs(raw_ias[i-1] - omega_from_rms[i])
        if disagreement > disagree_threshold:
            rms_g = np.exp(-(omega - omega_from_rms[i])**2 / (2*r_sigma**2))
            biased_pdf = pdf_map[:,i] * (1 + tracking_g * t_weight) * (1 + rms_g * r_weight)
        else:
            biased_pdf = pdf_map[:,i] * (1 + tracking_g * t_weight)

        raw_ias[i] = omega[np.argmax(biased_pdf)]

    should_be_zero = rms < rms_threshold
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

    pdf_map, omega = compute_pdf_map(Sxx_normalized, f, orders, min_omega, max_omega)

    plt.figure()
    plt.pcolormesh(t, omega, pdf_map)
    plt.show()

    ias = compute_ias(pdf_map, omega, Sxx_normalized, min_omega=min_omega, max_omega=max_omega)

    # Save IAS speed estimation to CSV file
    output_path = "ias_speed_estimation_rampenfunktion.csv"
    np.savetxt(output_path, np.column_stack((t, ias)), delimiter=',', header='time,ias', comments='')
    print(f"IAS speed estimation saved to {output_path}")

    plt.figure()
    plt.plot(t, ias)
    plt.show()