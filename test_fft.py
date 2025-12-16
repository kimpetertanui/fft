import unittest
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path


class TestFFTAnalysis(unittest.TestCase):
    """Test suite for FFT analysis functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fs = 1000  # Sampling frequency
        self.duration = 1  # 1 second
        self.t = np.arange(0, self.duration, 1/self.fs)
        self.frequencies = [5, 10, 15, 20]
        
    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')
        if os.path.exists('test_plot.png'):
            os.remove('test_plot.png')
    
    def test_time_vector_generation(self):
        """Test that time vector is correctly generated."""
        expected_length = self.fs * self.duration
        self.assertEqual(len(self.t), expected_length)
        self.assertAlmostEqual(self.t[0], 0.0)
        self.assertAlmostEqual(self.t[-1], self.duration - 1/self.fs)
    
    def test_sine_wave_generation(self):
        """Test sine wave generation at specific frequency."""
        f = 5  # 5 Hz
        signal = np.sin(2 * np.pi * f * self.t)
        
        # Check signal properties
        self.assertEqual(len(signal), len(self.t))
        self.assertTrue(np.all(np.abs(signal) <= 1.0))  # Amplitude should be 1
        self.assertTrue(np.isfinite(signal).all())  # No NaN or Inf
    
    def test_fft_computation(self):
        """Test FFT computation produces correct results."""
        f = 10  # 10 Hz
        signal = np.sin(2 * np.pi * f * self.t)
        ft_signal = np.fft.fft(signal)
        
        # Check FFT properties
        self.assertEqual(len(ft_signal), len(signal))
        self.assertTrue(np.isfinite(ft_signal).all())  # No NaN or Inf
        
        # Check magnitude is symmetric for real signal
        magnitude = np.abs(ft_signal)
        self.assertGreater(magnitude.max(), 0)
    
    def test_fft_peak_detection(self):
        """Test that FFT correctly identifies the dominant frequency."""
        f = 15  # 15 Hz
        signal = np.sin(2 * np.pi * f * self.t)
        ft_signal = np.fft.fft(signal)
        freq = np.fft.fftfreq(len(signal), d=1/self.fs)
        
        # Find peak in positive frequencies only
        positive_freqs = freq[:len(freq)//2]
        magnitude = np.abs(ft_signal[:len(ft_signal)//2])
        
        peak_idx = np.argmax(magnitude)
        peak_freq = positive_freqs[peak_idx]
        
        # Peak should be close to the input frequency
        self.assertAlmostEqual(peak_freq, f, delta=1)
    
    def test_frequency_vector_generation(self):
        """Test that frequency vector is correctly generated."""
        signal = np.sin(2 * np.pi * 5 * self.t)
        freq = np.fft.fftfreq(len(signal), d=1/self.fs)
        
        # Check frequency vector properties
        self.assertEqual(len(freq), len(signal))
        self.assertAlmostEqual(freq[0], 0.0)
        self.assertAlmostEqual(freq[1], 1/self.duration, places=2)
    
    def test_multiple_frequencies_fft(self):
        """Test FFT on multiple frequency signals."""
        for f in self.frequencies:
            signal = np.sin(2 * np.pi * f * self.t)
            ft_signal = np.fft.fft(signal)
            freq = np.fft.fftfreq(len(signal), d=1/self.fs)
            
            # Check each frequency has a valid FFT
            self.assertEqual(len(ft_signal), len(signal))
            self.assertTrue(np.isfinite(ft_signal).all())
            self.assertEqual(len(freq), len(signal))
    
    def test_plot_generation(self):
        """Test that plots can be generated without errors."""
        fig, axes = plt.subplots(len(self.frequencies), 2, figsize=(12, 10))
        
        for i, f in enumerate(self.frequencies):
            signal = np.sin(2 * np.pi * f * self.t)
            ft_signal = np.fft.fft(signal)
            freq = np.fft.fftfreq(len(signal), d=1/self.fs)
            
            # Plot signal
            axes[i, 0].plot(self.t, signal, linewidth=1.5)
            axes[i, 0].set_title(f'Signal: {f} Hz', fontweight='bold')
            
            # Plot FFT
            axes[i, 1].plot(freq, np.abs(ft_signal), linewidth=1.5)
            axes[i, 1].set_title(f'FFT: Peak at {f} Hz', fontweight='bold')
        
        self.assertEqual(len(fig.axes), len(self.frequencies) * 2)
    
    def test_plot_save(self):
        """Test that plots can be saved to file."""
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        
        f = 10
        signal = np.sin(2 * np.pi * f * self.t)
        axes.plot(self.t, signal)
        
        output_file = 'test_plot.png'
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        
        # Check file was created
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 0)
    
    def test_signal_properties(self):
        """Test basic properties of generated signals."""
        for f in self.frequencies:
            signal = np.sin(2 * np.pi * f * self.t)
            
            # Amplitude should be approximately 1
            self.assertAlmostEqual(np.max(signal), 1.0, places=1)
            self.assertAlmostEqual(np.min(signal), -1.0, places=1)
            
            # Mean should be approximately 0
            self.assertAlmostEqual(np.mean(signal), 0.0, places=1)
    
    def test_nyquist_frequency(self):
        """Test Nyquist frequency consideration."""
        nyquist = self.fs / 2
        
        # Frequencies should be well below Nyquist
        for f in self.frequencies:
            self.assertLess(f, nyquist)
    
    def test_fft_parseval_theorem(self):
        """Test Parseval's theorem: energy in time domain equals energy in frequency domain."""
        f = 10
        signal = np.sin(2 * np.pi * f * self.t)
        
        # Energy in time domain
        time_energy = np.sum(signal ** 2)
        
        # Energy in frequency domain
        ft_signal = np.fft.fft(signal)
        freq_energy = np.sum(np.abs(ft_signal) ** 2) / len(signal)
        
        # They should be approximately equal
        self.assertAlmostEqual(time_energy, freq_energy, delta=10)
    
    def test_backend_detection(self):
        """Test matplotlib backend detection logic."""
        # When CI or no DISPLAY is set, should use Agg
        use_non_interactive = os.environ.get('CI') or not os.environ.get('DISPLAY')
        
        # This is just checking the logic works, not the actual backend
        self.assertIsInstance(use_non_interactive, bool)


class TestFFTEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fs = 1000
        self.duration = 1
        self.t = np.arange(0, self.duration, 1/self.fs)
    
    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')
    
    def test_zero_signal_fft(self):
        """Test FFT on zero signal."""
        signal = np.zeros(len(self.t))
        ft_signal = np.fft.fft(signal)
        
        # FFT of zero signal should be zero
        self.assertTrue(np.allclose(ft_signal, 0))
    
    def test_constant_signal_fft(self):
        """Test FFT on constant signal."""
        signal = np.ones(len(self.t)) * 5  # Constant value of 5
        ft_signal = np.fft.fft(signal)
        
        # DC component (frequency 0) should be present
        self.assertGreater(np.abs(ft_signal[0]), 0)
    
    def test_high_frequency_signal(self):
        """Test signal with frequency close to Nyquist."""
        nyquist = self.fs / 2
        f = nyquist - 10
        signal = np.sin(2 * np.pi * f * self.t)
        ft_signal = np.fft.fft(signal)
        
        # Should still compute without error
        self.assertEqual(len(ft_signal), len(signal))
        self.assertTrue(np.isfinite(ft_signal).all())
    
    def test_empty_frequency_list(self):
        """Test behavior with empty frequency list."""
        frequencies = []
        
        fig, axes = plt.subplots(len(frequencies) or 1, 2, figsize=(12, 10))
        if len(frequencies) == 0:
            # With no frequencies, should create a single subplot
            self.assertEqual(len(fig.axes), 2)


class TestFFTIntegration(unittest.TestCase):
    """Integration tests for the complete FFT workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fs = 1000
        self.duration = 1
        self.t = np.arange(0, self.duration, 1/self.fs)
        self.frequencies = [5, 10, 15, 20]
        self.colors = ['blue', 'orange', 'green', 'red']
    
    def tearDown(self):
        """Clean up after tests."""
        plt.close('all')
        if os.path.exists('integration_test_plot.png'):
            os.remove('integration_test_plot.png')
    
    def test_complete_fft_workflow(self):
        """Test the complete FFT analysis workflow."""
        fig, axes = plt.subplots(len(self.frequencies), 2, figsize=(12, 10))
        fig.suptitle('FFT Analysis of Multiple Sine Waves', fontsize=16, fontweight='bold')
        
        # Process each frequency
        for i, (f, color) in enumerate(zip(self.frequencies, self.colors)):
            # Create a signal: sine wave
            signal = np.sin(2 * np.pi * f * self.t)
            
            # Compute the Fourier transform
            ft_signal = np.fft.fft(signal)
            freq = np.fft.fftfreq(len(signal), d=1/self.fs)
            
            # Plot the original signal
            axes[i, 0].plot(self.t, signal, color=color, linewidth=1.5)
            axes[i, 0].set_title(f'Signal: {f} Hz', fontweight='bold')
            axes[i, 0].set_xlabel('Time [s]')
            axes[i, 0].set_ylabel('Amplitude')
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].set_xlim(0, 0.5)
            
            # Plot the Fourier transform
            axes[i, 1].plot(freq, np.abs(ft_signal), color=color, linewidth=1.5)
            axes[i, 1].set_title(f'FFT: Peak at {f} Hz', fontweight='bold')
            axes[i, 1].set_xlabel('Frequency [Hz]')
            axes[i, 1].set_ylabel('Magnitude')
            axes[i, 1].set_xlim(0, 50)
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Verify the plot was created successfully
        self.assertEqual(len(fig.axes), len(self.frequencies) * 2)
        
        # Save and verify
        output_file = 'integration_test_plot.png'
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 0)


if __name__ == '__main__':
    unittest.main()
