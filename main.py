import numpy as np
from scipy.io import wavfile
from scipy import signal, fft
from numba import njit
from matplotlib import pyplot as plt

np.random.seed(0)

def save_wav(filename, data, sample_rate, base):
    # Convert float data to 16-bit PCM audio standard
    if (base <= 0):
        data_norm = np.int16((data / np.max(np.abs(data))) * 32767)
    else:
        data_norm = np.int16((data / base) * 32767)
    wavfile.write(filename, sample_rate, data_norm)
    print(f"Saved: {filename}")

# Generating the clean sample and analyzing statistics.
#w = np.linspace(0, np.pi/10, 3)
#x = sum([np.sin(w*n) for w in w])
filename = "../AudioData/ARCA23K.audio/429059.wav"
fs, raw_audio = wavfile.read(filename)
x = raw_audio / np.max(np.abs(raw_audio))
N = len(x)
X = fft.fft(x)
X = X[0:N//2]
w = np.linspace(0, np.pi, len(X))
plt.plot(w, 20*np.log10(np.abs(X)), linewidth=0.75)
plt.grid()
plt.title("Original Magnitude Spectrum")
plt.xlabel("Frequency rads/s")
plt.ylabel("Amplitude (dB)")
plt.show()
n = np.arange(N)

xStd = np.std(x)
xMean = np.mean(x)

print(f"The sample standard deviation of the audio sample is {xStd: .2f} volts.")
print(f"The average of the audio sample is {xMean: .2f} volts.\n")

# Generating the noisy signal.
e = np.random.normal(0, 1, N)
v = x + 0.25*e
spike_indices = np.random.randint(0, N, 1000)      # Pick 20 random moments in time
spike_amplitudes = np.random.choice([-15, 15], 1000) # Make them massive (+15 or -15 volts)
v[spike_indices] += spike_amplitudes             # Add them to the signal

# Analyzing the statistics of the noisy sample.
vStd = np.std(v)
vMean = np.mean(v)

print(f"The sample standard deviation of the noisy sample is {vStd: .2f} volts.")
print(f"The average of the noisy sample is {vMean: .2f} volts.\n")

# Replace samples that deviate too highly with the local average.
# Pro: Simple to implement and noises can be averaged out.
# Con: Local samples may also deviate too highly, so it is possible
#      that for some samples, the local average does nothing or increases
#      the deviation.
@njit
def statistical_filter(n0, data, tol, N):
    z = np.zeros(N)
    for i in range(0, N):
        slice = data[i:i+n0]
        slice = np.concat((slice, data[i+n0+1:i+2*n0+1]))
        mean_local = np.mean(slice)
        std_local = np.std(slice)
        if (data[i+n0] >= mean_local + tol*std_local or data[i+n0] <= mean_local - tol*std_local):
            z[i] = (np.sum(data[i:i+2*n0+1]) - data[i+n0])/(2*n0)
        else:
            z[i] = data[i + n0]

    return z

n0 = 25
v_padded = np.pad(v, (n0, n0), "constant", constant_values=(0, 0))
z = statistical_filter(25, v_padded, 3, N)

# Analyzing statistics of filtered sample.
zStd = np.std(z)
zMean = np.mean(z)

print(f"The sample standard deviation of the statistically filtered sample is {zStd: .2f} volts.")
print(f"The average of the statistically filtered sample is {zMean: .2f} volts.\n")

# creating and applying a low-pass FIR filter
W = 1001 # window length of W
wc = np.pi/3 # cutoff frequency
n_windowed = np.arange(-W//2, W//2 + 1)
h = ((np.cos(np.pi * n_windowed/(W//2)) + 1)/2) * (wc / np.pi) * np.sinc((wc / np.pi) * n_windowed)

y = signal.convolve(z, h, "same")

# Analyzing statistics of filtered sample.
yStd = np.std(y)
yMean = np.mean(y)

print(f"The sample standard deviation of the filtered sample is {yStd: .2f} volts.")
print(f"The average of the filtered sample is {yMean: .2f} volts.\n")

# we do the same for the noisy sample to compare results later
y_noStats = signal.convolve(v, h, "same")

yNoStatsStd = np.std(y_noStats)
yNoStatsMean = np.mean(y_noStats)

print(f"The sample standard deviation of the filtered sample without statistical filtering is {yNoStatsStd: .2f} volts.")
print(f"The average of the filtered sample without statistical filtering is {yNoStatsMean: .2f} volts.\n")

# we then apply a four-sample local average filter in order to
# further reduce the effect of the Guassian noise
# y[n] = (x[n] + x[n-2])/2
y_avg = signal.convolve(y, [0.5, 0, 0.5], "same")

yAvgStd = np.std(y_avg)
yAvgMean = np.mean(y_avg)

print(f"The sample standard deviation of the averaged filtered sample is {yAvgStd: .2f} volts.")
print(f"The average of the averaged filtered sample is {yAvgMean: .2f} volts.\n")

# calculating normalized root mean square error
v_rmse = np.sqrt(np.mean((x - v)**2))
z_rmse = np.sqrt(np.mean((x - z)**2))
y_rmse = np.sqrt(np.mean((x - y)**2))
y_noStats_rmse = np.sqrt(np.mean((x - y_noStats)**2))
y_avg_rmse = np.sqrt(np.mean((x - y_avg)**2))
signal_range = np.max(x) - np.min(x)
vError = v_rmse / signal_range
zError = z_rmse / signal_range
yError = y_rmse / signal_range
yNoStatsError = y_noStats_rmse / signal_range
yAvgError = y_avg_rmse / signal_range

print(f"The normalized root mean square error of the dirty audio is {vError:.2f}.")
print(f"The normalized root mean square error of the statistically filtered audio is {zError:.2f}.")
print(f"The normalized root mean square error of the filtered audio is {yError:.2f}.")
print(f"The normalized root mean square error of the averaged filtered audio is {yAvgError:.2f}.")
print(f"The normalized root mean square error of the filtered audio without statistical filtering is {yNoStatsError:.2f}.\n")

# calculating the SINR using fft
x_power = (1/N**2)*np.sum(np.abs(fft.fft(x))**2)
v_noise_power = (1/N**2)*np.sum(np.abs(fft.fft(v-x))**2)
z_noise_power = (1/N**2)*np.sum(np.abs(fft.fft(z-x))**2)

No = len(y[W//2:-W//2])
xo_power = (1/No**2)*np.sum(np.abs(fft.fft(x[W//2:-W//2]))**2)
y_noise_power = (1/No**2)*np.sum(np.abs(fft.fft(y[W//2:-W//2]-x[W//2:-W//2]))**2)
y_noStats_noise_power = (1/No**2)*np.sum(np.abs(fft.fft(y_noStats[W//2:-W//2]-x[W//2:-W//2]))**2)

N1 = len(y[W//2+1:-W//2-1])
x1_power = (1/N1**2)*np.sum(np.abs(fft.fft(x[W//2+1:-W//2-1]))**2)
y_avg_noise_power = (1/N1**2)*np.sum(np.abs(fft.fft(y_avg[W//2+1:-W//2-1]-x[W//2+1:-W//2-1]))**2)

SINR_v = 10*np.log10(x_power / v_noise_power)
SINR_z = 10*np.log10(x_power / z_noise_power)
SINR_y = 10*np.log10(xo_power / y_noise_power)
SINR_y_noStats = 10*np.log10(xo_power / y_noStats_noise_power)
SINR_y_avg = 10*np.log10(x1_power / y_avg_noise_power)

print(f"The SINR of the dirty audio is {SINR_v:.2f} dB.")
print(f"The SINR of the statistically filtered audio is {SINR_z:.2f} dB.")
print(f"The SINR of the filtered audio is {SINR_y:.2f} dB.")
print(f"The SINR of the averaged filtered audio is {SINR_y_avg:.2f} dB.")
print(f"The SINR of the filtered audio without statistical filtering is {SINR_y_noStats:.2f} dB\n.")

base = np.max((x, v, z, y, y_noStats, y_avg))
save_wav("original.wav", x, fs, base)
save_wav("noisy.wav", v, fs, base)
save_wav("stat_filtered.wav", z, fs, base)
save_wav("filtered.wav", y, fs, base)
save_wav("filtered_noStat.wav", y_noStats, fs, base)
save_wav("filtered_avg.wav", y_avg, fs, base)
import numpy as np
from scipy.io import wavfile
from scipy import signal, fft
from numba import njit
from matplotlib import pyplot as plt

np.random.seed(0)

def save_wav(filename, data, sample_rate):
    # Convert float data to 16-bit PCM audio standard
    data_norm = np.int16((data / np.max(np.abs(data))) * 32767)
    wavfile.write(filename, sample_rate, data_norm)
    print(f"Saved: {filename}")

# Generating the clean sample and analyzing statistics.
#w = np.linspace(0, np.pi/10, 3)
#x = sum([np.sin(w*n) for w in w])
filename = "../AudioData/ARCA23K.audio/429059.wav"
fs, raw_audio = wavfile.read(filename)
x = raw_audio / np.max(np.abs(raw_audio))
N = len(x)
X = fft.fft(x)
X = X[0:N//2]
w = np.linspace(0, np.pi, len(X))
plt.plot(w, 20*np.log10(np.abs(X)), linewidth=0.75)
plt.grid()
plt.title("Original Magnitude Spectrum")
plt.xlabel("Frequency rads/s")
plt.ylabel("Amplitude (dB)")
plt.show()
n = np.arange(N)

xStd = np.std(x)
xMean = np.mean(x)

print(f"The sample standard deviation of the audio sample is {xStd: .2f} volts.")
print(f"The average of the audio sample is {xMean: .2f} volts.\n")

# Generating the noisy signal.
e = np.random.normal(0, 1, N)
v = x + 0.25*e
spike_indices = np.random.randint(0, N, 1000)      # Pick 20 random moments in time
spike_amplitudes = np.random.choice([-15, 15], 1000) # Make them massive (+15 or -15 volts)
v[spike_indices] += spike_amplitudes             # Add them to the signal

# Analyzing the statistics of the noisy sample.
vStd = np.std(v)
vMean = np.mean(v)

print(f"The sample standard deviation of the noisy sample is {vStd: .2f} volts.")
print(f"The average of the noisy sample is {vMean: .2f} volts.\n")

# Replace samples that deviate too highly with the local average.
# Pro: Simple to implement and noises can be averaged out.
# Con: Local samples may also deviate too highly, so it is possible
#      that for some samples, the local average does nothing or increases
#      the deviation.
@njit
def statistical_filter(n0, data, tol, N):
    z = np.zeros(N)
    for i in range(0, N):
        slice = data[i:i+n0]
        slice = np.concat((slice, data[i+n0+1:i+2*n0+1]))
        mean_local = np.mean(slice)
        std_local = np.std(slice)
        if (data[i+n0] >= mean_local + tol*std_local or data[i+n0] <= mean_local - tol*std_local):
            z[i] = (np.sum(data[i:i+2*n0+1]) - data[i+n0])/(2*n0)
        else:
            z[i] = data[i + n0]

    return z

n0 = 25
v_padded = np.pad(v, (n0, n0), "constant", constant_values=(0, 0))
z = statistical_filter(25, v_padded, 3, N)

# Analyzing statistics of filtered sample.
zStd = np.std(z)
zMean = np.mean(z)

print(f"The sample standard deviation of the statistically filtered sample is {zStd: .2f} volts.")
print(f"The average of the statistically filtered sample is {zMean: .2f} volts.\n")

# creating and applying a low-pass FIR filter
W = 1001 # window length of W
wc = np.pi/3 # cutoff frequency
n_windowed = np.arange(-W//2, W//2 + 1)
h = ((np.cos(np.pi * n_windowed/(W//2)) + 1)/2) * (wc / np.pi) * np.sinc((wc / np.pi) * n_windowed)

y = signal.convolve(z, h, "same")

# Analyzing statistics of filtered sample.
yStd = np.std(y)
yMean = np.mean(y)

print(f"The sample standard deviation of the filtered sample is {yStd: .2f} volts.")
print(f"The average of the filtered sample is {yMean: .2f} volts.\n")

# we do the same for the noisy sample to compare results later
y_noStats = signal.convolve(v, h, "same")

yNoStatsStd = np.std(y_noStats)
yNoStatsMean = np.mean(y_noStats)

print(f"The sample standard deviation of the filtered sample without statistical filtering is {yNoStatsStd: .2f} volts.")
print(f"The average of the filtered sample without statistical filtering is {yNoStatsMean: .2f} volts.\n")

# we then apply a four-sample local average filter in order to
# further reduce the effect of the Guassian noise
# y[n] = (x[n] + x[n-2])/2
y_avg = signal.convolve(y, [0.5, 0, 0.5], "same")

yAvgStd = np.std(y_avg)
yAvgMean = np.mean(y_avg)

print(f"The sample standard deviation of the averaged filtered sample is {yAvgStd: .2f} volts.")
print(f"The average of the averaged filtered sample is {yAvgMean: .2f} volts.\n")

# calculating normalized root mean square error
v_rmse = np.sqrt(np.mean((x - v)**2))
z_rmse = np.sqrt(np.mean((x - z)**2))
y_rmse = np.sqrt(np.mean((x - y)**2))
y_noStats_rmse = np.sqrt(np.mean((x - y_noStats)**2))
y_avg_rmse = np.sqrt(np.mean((x - y_avg)**2))
signal_range = np.max(x) - np.min(x)
vError = v_rmse / signal_range
zError = z_rmse / signal_range
yError = y_rmse / signal_range
yNoStatsError = y_noStats_rmse / signal_range
yAvgError = y_avg_rmse / signal_range

print(f"The normalized root mean square error of the dirty audio is {vError:.2f}.")
print(f"The normalized root mean square error of the statistically filtered audio is {zError:.2f}.")
print(f"The normalized root mean square error of the filtered audio is {yError:.2f}.")
print(f"The normalized root mean square error of the averaged filtered audio is {yAvgError:.2f}.")
print(f"The normalized root mean square error of the filtered audio without statistical filtering is {yNoStatsError:.2f}.\n")



# calculating the SINR using fft
x_power = (1/N**2)*np.sum(np.abs(fft.fft(x))**2)
v_noise_power = (1/N**2)*np.sum(np.abs(fft.fft(v-x))**2)
z_noise_power = (1/N**2)*np.sum(np.abs(fft.fft(z-x))**2)

No = len(y[W//2:-W//2])
xo_power = (1/No**2)*np.sum(np.abs(fft.fft(x[W//2:-W//2]))**2)
y_noise_power = (1/No**2)*np.sum(np.abs(fft.fft(y[W//2:-W//2]-x[W//2:-W//2]))**2)
y_noStats_noise_power = (1/No**2)*np.sum(np.abs(fft.fft(y_noStats[W//2:-W//2]-x[W//2:-W//2]))**2)

N1 = len(y[W//2+1:-W//2-1])
x1_power = (1/N1**2)*np.sum(np.abs(fft.fft(x[W//2+1:-W//2-1]))**2)
y_avg_noise_power = (1/N1**2)*np.sum(np.abs(fft.fft(y_avg[W//2+1:-W//2-1]-x[W//2+1:-W//2-1]))**2)

SINR_v = 10*np.log10(x_power / v_noise_power)
SINR_z = 10*np.log10(x_power / z_noise_power)
SINR_y = 10*np.log10(xo_power / y_noise_power)
SINR_y_noStats = 10*np.log10(xo_power / y_noStats_noise_power)
SINR_y_avg = 10*np.log10(x1_power / y_avg_noise_power)

print(f"The SINR of the dirty audio is {SINR_v:.2f} dB.")
print(f"The SINR of the statistically filtered audio is {SINR_z:.2f} dB.")
print(f"The SINR of the filtered audio is {SINR_y:.2f} dB.")
print(f"The SINR of the averaged filtered audio is {SINR_y_avg:.2f} dB.")
print(f"The SINR of the filtered audio without statistical filtering is {SINR_y_noStats:.2f} dB\n.")

save_wav("original.wav", x, fs)
save_wav("noisy.wav", v, fs)
save_wav("stat_filtered.wav", z, fs)
save_wav("filtered.wav", y, fs)
save_wav("filtered_noStat.wav", y_noStats, fs)
save_wav("filtered_avg.wav", y_avg, fs)
