import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys

# Use non-interactive backend if running in CI or no display available
if os.environ.get('CI') or not os.environ.get('DISPLAY', sys.platform == 'darwin'):
    matplotlib.use('Agg')  # Non-interactive backend for headless environments

# Parameters
fs = 1000  # Sampling frequency
t = np.arange(0, 1, 1/fs)  # Time vector from 0 to 1 second in 1/fs increments

# Define different frequencies for comparison
frequencies = [5, 10, 15, 20]  # Different sine wave frequencies
colors = ['blue', 'orange', 'green', 'red']

# Create a figure with subplots
fig, axes = plt.subplots(len(frequencies), 2, figsize=(12, 10))
fig.suptitle('FFT Analysis of Multiple Sine Waves', fontsize=16, fontweight='bold')

# Process each frequency
for i, (f, color) in enumerate(zip(frequencies, colors)):
    # Create a signal: sine wave
    signal = np.sin(2 * np.pi * f * t)
    
    # Compute the Fourier transform
    ft_signal = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal), d=1/fs)  # Frequency vector
    
    # Plot the original signal
    axes[i, 0].plot(t, signal, color=color, linewidth=1.5)
    axes[i, 0].set_title(f'Signal: {f} Hz', fontweight='bold')
    axes[i, 0].set_xlabel('Time [s]')
    axes[i, 0].set_ylabel('Amplitude')
    axes[i, 0].grid(True, alpha=0.3)
    axes[i, 0].set_xlim(0, 0.5)  # Show only first 0.5 seconds for clarity
    
    # Plot the Fourier transform
    axes[i, 1].plot(freq, np.abs(ft_signal), color=color, linewidth=1.5)
    axes[i, 1].set_title(f'FFT: Peak at {f} Hz', fontweight='bold')
    axes[i, 1].set_xlabel('Frequency [Hz]')
    axes[i, 1].set_ylabel('Magnitude')
    axes[i, 1].set_xlim(0, 50)  # Limit frequency range for better visibility
    axes[i, 1].grid(True, alpha=0.3)

plt.tight_layout()

# Save plot if in CI environment, otherwise show it
if os.environ.get('CI'):
    output_file = 'fft_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ FFT analysis completed successfully!")
    print(f"📊 Plot saved to: {output_file}")
else:
    plt.show()    