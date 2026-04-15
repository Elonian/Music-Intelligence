# Music Intelligence

## Abstract

This project presents two complementary music intelligence pipelines centered on symbolic musical data. The first pipeline performs note driven audio synthesis by converting symbolic note sequences into sampled sine and sawtooth waveforms, then applying fade, delay, and layered mixing operations to analyze how waveform structure, amplitude evolution, and harmonic content change over time. The second pipeline performs symbolic MIDI classification by extracting compact pitch, timing, velocity, and channel-based features and using them to separate piano files from drum patterns with a standardized logistic regression model. Together, these components frame music as both a signal-generation problem and a symbolic pattern-recognition problem, while providing a reproducible project structure for visualization, comparison, and evaluation.

## Output Gallery

### Audio Synthesis

![Audio Animated Panel](outputs/sine_wave_binary_classification/readme/readme_audio_animated_panel.gif)

The synthesis summary follows one melody through pure sinusoidal rendering, harmonic enrichment, amplitude decay, delay, and layering so the waveform envelope and spectral content can be read together.

### Symbolic Classification

![Classifier Animated Panel](outputs/sine_wave_binary_classification/readme/readme_classifier_animated_panel.gif)

The classification summary shows where `piano` and `drums` separate in symbolic feature space, how the full feature vector differs between the two classes, and why the expanded descriptor remains stable across alternate train/test splits.

## Setup

From the project root:

```bash
cd /mntdatalora/src/Music-Intelligence
pip install -r requirements.txt
```

Core dependencies:

- `numpy`
- `scipy`
- `mido`
- `scikit-learn`
- `matplotlib`
- `imageio`
- `Pillow`
- `nbformat`

## Data Layout

Input bundle:

```text
data/
  sine_wave_binary_classification/
    homework1/
      homework1_stub.ipynb
      input.wav
      output.wav
      piano.zip
      drums.zip
      piano/
      drums/
```

Generated outputs:

```text
outputs/
  sine_wave_binary_classification/
    sine_wave/
    binary_classification/
    visuals/
      audio/
      classifier/
    readme/
    evaluation/
```

The MIDI utilities search the provided data bundle first, automatically extract `piano.zip` and `drums.zip` when needed, and then write project outputs into separate directories for rendered audio, classifier artifacts, raw visuals, README-ready panels, and compact evaluation summaries.

## Execution Order

Typical end-to-end run:

```bash
python scripts/sine_wave/build_audio_gallery.py
python scripts/visualiser/render_audio_gallery.py
python scripts/binary_classify/train_midi_classifier.py --max-files 120
python scripts/visualiser/render_classifier_gallery.py --max-files 120
python evaluation/compute_metrics.py
python scripts/build_readme_panels.py
```

Optional:

```bash
python scripts/visualiser/render_evaluation_gallery.py --max-files 120
```

## Audio Synthesis

### Model

For a melody represented as an ordered note-duration sequence

```math
\mathcal{M} = \{(m_i, d_i)\}_{i=1}^{L},
```

the note-frequency conversion follows the equal-tempered relation

```math
f(m) = 440 \cdot 2^{\frac{m - 69}{12}},
```

where `m` is the MIDI note number implied by the note name. The base sine renderer for one note is

```math
x_i(t) = \sin(2 \pi f(m_i) t), \qquad 0 \le t < d_i,
```

and the sawtooth approximation adds 18 upper harmonics:

```math
x^{(i)}_{\text{saw}}(t) = \frac{2}{\pi} \sum_{k=1}^{19} \frac{(-1)^{k+1}}{k} \sin(2 \pi k f(m_i) t).
```

The full melody is created by time concatenation

```math
x_{\text{melody}} = x_1 \oplus x_2 \oplus \cdots \oplus x_L,
```

and the effect stage applies a linear fade, a discrete delay, and a weighted simultaneous mix:

```math
x_{\text{fade}}[n] = \left(1 - \frac{n}{N-1}\right) x[n],
```

```math
x_{\text{delay}}[n] = x[n] + \alpha x[n-d]\mathbf{1}[n \ge d],
```

```math
x_{\text{mix}}[n] = \sum_{i=1}^{K} g_i x_i[n].
```

With sample rate `f_s` and delay time `\tau`, the offset is `d = \lfloor \tau f_s \rfloor`. The audio section is therefore organized around waveform, envelope, and spectrogram views because the pipeline changes both amplitude over time and harmonic energy across frequency.

### Static Panel

![Audio Static Panel](outputs/sine_wave_binary_classification/readme/readme_audio_static_panel.png)

### Current Metrics

| Artifact | Duration [s] | Role |
| --- | ---: | --- |
| `melody_sine.wav` | `3.65` | base melody rendered with sine waves |
| `melody_sawtooth.wav` | `3.65` | harmonic melody rendered with the sawtooth series |
| `melody_faded.wav` | `3.65` | linearly decayed melody |
| `melody_delayed.wav` | `4.15` | original melody plus `0.50 s` echo tail |
| `melody_stacked.wav` | `3.65` | simultaneous layered mix of lead and pad voices |

## Symbolic Classification

### Model

Each MIDI file is summarized into symbolic note statistics rather than raw audio features. The baseline vector is

```math
x_{\text{base}} = [p_{\min},\; p_{\max},\; n_{\text{unique}},\; \bar{p}],
```

and the expanded vector is

```math
x_{\text{enh}} = [p_{\min},\; p_{\max},\; n_{\text{unique}},\; \bar{p},\; p_{\text{span}},\; \log(1+b),\; \log(1+\rho),\; \bar{v}/127,\; r_{\text{drum}}],
```

with

```math
p_{\text{span}} = p_{\max} - p_{\min}, \qquad
b = \frac{T_{\max}}{\text{ticks per beat}}, \qquad
\rho = \frac{N}{b}, \qquad
r_{\text{drum}} = \frac{N_{\text{channel 9}}}{N}.
```

The standardized classifier is

```math
\tilde{x}_j = \frac{x_j - \mu_j}{\sigma_j},
```

followed by logistic regression:

```math
z = w^\top \tilde{x} + \beta, \qquad P(y = 1 \mid x) = \sigma(z),
```

with decision rule

```math
\hat{y} =
\begin{cases}
1 & \text{if } z > 0 \\
0 & \text{otherwise}
\end{cases}
```

where `y = 1` denotes `piano` and `y = 0` denotes `drums`.

Feature definitions:

| Feature | Meaning | Why it helps |
| --- | --- | --- |
| `lowest_pitch`, `highest_pitch` | minimum and maximum active note numbers | separates narrow drum pitch sets from wider piano ranges |
| `unique_pitch_num` | number of distinct note values | captures pitch diversity |
| `average_pitch_value` | mean of unique active pitches | shifts piano files toward tonal centers |
| `pitch_span` | `highest_pitch - lowest_pitch` | measures melodic range |
| `log_beats` | `log(1 + beat_count)` | normalizes long symbolic sequences |
| `log_note_density` | `log(1 + note_count / beat_count)` | captures event density per beat |
| `average_velocity_norm` | mean velocity divided by `127` | reflects attack intensity |
| `drum_channel_ratio` | fraction of active notes on MIDI channel `9` | strong drum-specific structural cue |

The panel combines low-dimensional scatter views with the full nine-feature profile so the separation is not reduced to a single score.

### Static Panel

![Classifier Static Panel](outputs/sine_wave_binary_classification/readme/readme_classifier_static_panel.png)

### Current Metrics

| Evaluation | Baseline | Enhanced | Interpretation |
| --- | ---: | ---: | --- |
| fixed split (`random_state = 42`) | `1.000` | `1.000` | both models separate the saved split perfectly |
| 8-seed sweep mean | `0.964` | `1.000` | the enhanced vector stays saturated across alternate train/test splits |
| 8-seed sweep minimum | `0.917` | `1.000` | the baseline drops on harder splits while the enhanced model does not |

So the apparent `1.0 / 1.0` tie in the saved confusion matrices is real for that specific split, but it is not the whole story. The expanded timing, velocity, and drum-channel features improve robustness once the train/test partition changes.

## Evaluation

The evaluation folder stays table-first:

| Metric | Value |
| --- | ---: |
| `audio_clip_count` | `5` |
| `lead_note_count` | `9` |
| `sample_rate` | `44100` |
| `delay_tail_seconds` | `0.5000` |
| `baseline_accuracy` | `1.0000` |
| `enhanced_accuracy` | `1.0000` |
| `baseline_seed_sweep_mean` | `0.9635` |
| `enhanced_seed_sweep_mean` | `1.0000` |
| `baseline_seed_sweep_min` | `0.9167` |
| `enhanced_seed_sweep_min` | `1.0000` |
| `row_count` | `240` |
