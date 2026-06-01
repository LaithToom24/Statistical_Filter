# Audio Denoising — Spike Removal + FIR Low-Pass Pipeline

A small signal-processing project where I take a clean audio recording, deliberately
wreck it with two very different kinds of noise, and then try to get it back. The point
wasn't just to "make it sound better" — it was to build each filtering stage around the
*specific* kind of noise it's meant to deal with, and then actually measure whether each
stage earns its place.

## The idea

Real recordings get corrupted in more than one way at once, so I model two noise types
that need completely different treatment:

- **Gaussian (background) noise** — a steady hiss spread across the whole signal. I add
  this as `0.25 * e`, where `e` is standard normal noise.
- **Impulsive spikes** — sudden, huge clicks/pops (think a bad cable or a dropped sample).
  I add 1000 of these at random positions, each one a massive ±15 against a signal whose
  amplitude is around 1.

These two are worth separating because **a linear filter can clean up the hiss but is
hopeless against the spikes**. A spike is broadband, so a chunk of its energy sits right
inside any passband you choose, and worse, a low-pass filter smears each spike out into a
long ringing tail. So the spikes have to come off *first*, with something nonlinear,
before any smoothing happens.

That ordering is the whole thesis of the project, and the results below back it up.

## How the pipeline works

The signal goes through three stages, in this order:

### 1. Statistical (outlier) filter — kills the spikes

This slides a window across the signal and, for each sample, compares it to the mean and
standard deviation of its neighbours. If the sample sits more than `tol` standard
deviations away from the local mean, I treat it as a spike and replace it with the average
of the surrounding samples. Otherwise I leave it alone.

One detail that matters: when I compute the local mean and std, I **exclude the sample
being tested**. If you leave it in, a giant spike inflates the very standard deviation
you're using to judge it, which deflates its own outlier score and lets it slip through —
especially at higher thresholds. Leaving it out means each spike gets measured against
clean neighbours and stands out enormously, so I can use a comfortable `tol = 3` and still
catch essentially every spike without chewing up the real signal.

### 2. Low-pass FIR filter — handles the Gaussian hiss

Once the spikes are gone, the leftover problem is the broadband Gaussian noise. I knock it
down with a windowed-sinc low-pass FIR filter: an ideal sinc with cutoff `wc = π/3`,
multiplied by a Hann window (length 1001) to tame the ringing you'd otherwise get from
truncating the sinc. Everything above the cutoff — including most of the noise power — gets
attenuated.

### 3. Light averaging filter — a final smooth

A small two-tap average (`(y[n-1] + y[n+1]) / 2`) as a final pass to shave off a bit more
high-frequency residual. It's a gentle extra smoothing step on top of the FIR.

## Measuring whether it worked

I didn't want to judge this by ear alone, so I track two metrics for every stage:

- **NRMSE** — root-mean-square error against the original clean signal, normalised by the
  signal's range. Lower is better.
- **SINR** — signal-to-interference-plus-noise ratio, computed in the frequency domain via
  the FFT (Parseval), as `10 * log10(signal_power / residual_noise_power)`. Higher is
  better. Where a filter introduces edge transients I trim the contaminated samples off
  both ends before measuring, so the SINR reflects the steady-state result rather than the
  start-up ringing.

## Results

Representative run (seed fixed at 0):

| Stage | SINR (dB) | What it tells me |
|---|---|---|
| Dirty signal | ≈ −6.7 | Starting point — spikes dominate the noise power |
| Low-pass only (no spike removal) | ≈ −2.3 | A linear filter alone barely helps |
| Spike removal only | ≈ +1.0 | Removing the spikes is the single biggest win |
| Spike removal + low-pass | ≈ +4.1 | Now the hiss gets cleaned up too |
| + averaging | ≈ +5.3 | A final small gain |

The line I find most satisfying is the second one. **Throwing only a low-pass filter at the
problem leaves it worse than just removing the spikes** — and far worse than doing both in
the right order. That gap is the entire justification for the nonlinear first stage: a
linear filter cannot deal with impulse noise, full stop.

(SINR values come from one run. Because the cleaner signals have so little residual noise,
their SINR estimate is fairly sensitive to the exact noise realisation, which is why I fix
the random seed — so the numbers reflect the code, not the luck of the draw.)

## Running it

You'll need Python 3 and:

```
pip install numpy scipy numba matplotlib
```

Then point the script at a `.wav` file by editing the `filename` line near the top
(it's currently set to a file from the ARCA23K dataset, so you'll want to change it to a
file you actually have):

```python
filename = "../AudioData/ARCA23K.audio/429059.wav"
```

and run:

```
python main.py
```

It will print the statistics, NRMSE and SINR for every stage, show the magnitude spectrum
of the original, and write out the audio at each step.

## What it produces

The script saves a `.wav` for every stage so you can listen to the difference:

| File | What it is |
|---|---|
| `original.wav` | the clean reference |
| `noisy.wav` | after adding Gaussian noise + spikes |
| `stat_filtered.wav` | after spike removal only |
| `filtered.wav` | after spike removal + low-pass |
| `filtered_avg.wav` | after the final averaging pass |
| `filtered_noStat.wav` | low-pass only, no spike removal (the control) |

## Things I know could be better

This is a learning project, and there are a few rough edges I'm aware of:

- **Edge handling.** I zero-pad for the statistical filter and use `"same"` convolutions,
  which leaves transients at the very start and end of the signal. I trim those out before
  measuring SINR, but the saved audio still carries them.
- **The averaging stage is a trade-off.** It improves SINR on paper, but its frequency
  response droops at the top of the passband, so it can make things sound slightly duller.
  Worth an A/B listen between `filtered.wav` and `filtered_avg.wav` to decide if it's
  actually the better result.
- **Global normalisation.** Since I used the maximum value between all of the sequences in
  order to normalize the data, you might need to increase your volume in order to hear some
  of the audio files.
- **Going further.** The remaining noise is Gaussian hiss sitting *inside* the passband,
  and no fixed linear filter can remove that without taking signal with it. The natural
  next step would be signal-adaptive denoising — Wiener filtering, spectral subtraction, or
  a wavelet approach — at the cost of more complexity and their own artefacts.

## Notes

The test audio comes from the [ARCA23K](https://zenodo.org/records/5117901) sound-event
dataset, but any mono `.wav` will work.
