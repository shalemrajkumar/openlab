## Signal Processing

- Data Aquisition
- Signal conditioning
- Feature extraction
- Hypothesis testing


### Data Aquisition

- Recording the signal with maximizing SNR
- Sampling the analog signal to digital signal. without losing information

#### analog to digital conversion

context:

the signal is recored a continuous electronic (voltage) signal. the signal is sampled and quantized to a finite set of values.

- Sampling
    - The process of converting a continuous signal into a discrete signal by taking samples at regular intervals.
    - $x[n] = x(nT_s)$
    - where $ T_s $ is the sampling interval and $ n $ is the sample number.

- Quantization

    - The process of mapping a sampled points to a finite range of discrete values (Amplitudes).
    - Input: You sample a signal at fixed time steps, getting [0.37V, -0.82V, 1.12V].
    
    > Quantization:

  
        `` Assume a range of -1V to +1V and 3-bit quantization (8 levels).
            Step size: 2/8 = 0.25V.
            Levels: [−1,−0.75,−0.5,−0.25,0,0.25,0.5,0.75]
             Map each sample:
                0.37V → 0.5V
                -0.82V → -0.75V
                1.12V → 0.75V (clipped to max)
    
    - Output: [0.5V, -0.75V, 0.75V]—now discrete in both time and amplitude.
    - These values are then stored as binary numbers.
        - For example, 0.5V might be stored as 010 in binary.
    - Quantization error: The difference between the actual signal and the quantized value.
    
- `convention`
    - T_s : sampling period
    - F_s : sampling frequency

- Sampling causes loss of information

- Example:
    - $$x[n] = a \cos(2\pi f_0 n T_s + \phi)$$
    - what could be the original x(t) if sampled at T_s?
        - may be a $x(t) = a\cos(2\pi f_0 n t + \phi)$, but not necessarily.
        - $x(t) = a \cos(2\pi (f_0 + mF_s)t + \phi)$
        - there are infinitely many possible x(t) that could produce the same samples.
    


#### frequencie domain and time domain representations

- frequency domain and time domain representation

    - $$X(f) = \sum_{n=-\infty}^{\infty} x[n] e^{-j2\pi fnT_s}$$
    - where $X(f)$ is the Fourier transform of the discrete signal $x[n]$.

    altertively
    - $$x(t) = \sum_{n=-\infty}^{\infty} x[n] e^{j2\pi fnT_s}$$

    where X(f) is the spectrum of the x(t) signal.


#### Nyquist theorem

- The Nyquist-Shannon sampling theorem 
    - $f_s >= 2f_{max}$
    - where $f_{max}$ is the maximum frequency of the signal.
    - If the sampling rate is too low, aliasing occurs, causing high-frequency components to be misrepresented as lower frequencies.

#### Nyquist interpolation formula
   
- $$x(t) = \sum_{n=-\infty}^{\infty} x[n] \text{sinc}(f_s(t - nT_s))$$
    - Using the sinc function as the interpolation kernel. its simple scaling and shifting

    - why sinc function ?
        - its zero-crossings at integer multiples of T_s
        - its integral over one period is 1
        - $x(mT_s)$ = $\sum {n=-\infty}^{\infty} x[n] \text{sinc}((m - n)) = x[m]$ since sinc is 0 everywhere 
        - its Fourier transform is a rectangular function
        - there are many ways to interpulate like rectangular, triangular, and hamming window functions.

- 


- $x(t) \rightarrow sampling \as \fract{1}{T_s} \rightarrow x[n] \rightarrow quantization \rightarrow x_q[n]$
- $x[n] \rightarrow \sinc \interpolation \rightarrow x(t)$


### Digital signal filtering

- filtering certain frequencies

#### Linear constant-coefficient difference equation (LCCD)

y[n] = \sum_{k=1}^{N} a_k y[n-k] + \sum_{m=0}^{M} b_m x[n-m]

- where $y[n]$ is the output signal, $x[n]$ is the input signal, $a_k$ and $b_m$ are the filter coefficients, and N and M are the filter orders.

- Example :

    - Amplifier : y[n] = Gain * x[n]
    - Delay : y[n] = x[n - n_0]
    - Two point moving average filter : y[n] = 0.5 * (x[n] + x[n - 1])
    - Low pass filter : y[n] = 0.5 * (x[n] + x[n - 1]) - 0.5 * y[n - 1]
    - High pass filter : y[n] = x[n] - 0.5 * (x[n] + x[n - 1]) - 0.5 * y[n - 1]
    - Band pass filter : y[n] = 0.5 * (x[n] + x[n - 1]) - 0.5 * y[n - 1] + 0.5 * (x[n] - x[n - 1]) - 0.5 * y[n - 1]
    - Band stop filter : y[n] = 0.5 * (x[n] + x[n - 1]) - 0.5 * y[n - 1] + 0.5 * (x[n] - x[n - 1]) - 0.5 * y[n - 1]
    - Notch filter : y[n] = 0.5 * (x[n] + x[n - 1]) - 0.5 * y[n - 1] + 0.5 * (x[n] - x[n - 1]) - 0.5 * y[n - 1]
    - Comb filter : y[n] = x[n] - 0.5 * (x[n] + x[n - 1]) - 0.5 * y[n - 1] + 0.5 * (x[n] - x[n - 1]) - 0.5 * y[n - 1]
    - Adaptive filter : y[n] = x[n] - 0.5 * (x[n] + x[n - 1]) - 0.5 * y[n - 1] + 0.5 * (x[n] - x[n - 1]) - 0.5 * y[n - 1]
    - Linear phase filter : y[n] = x[n] - 0.5 * (x[n] + x[n - 1]) - 0.5 * y[n - 1] + 0.5 * (x[n] - x[n - 1]) - 0.5 * y[n - 1]
    - FIR filter : y[n] = \sum_{k=0}^{N} b_k x[n-k]
    - IIR filter : y[n] = \sum_{k=0}^{N} b_k x[n-k] + \sum_{m=1}^{M} a_m y[n-m]
    - Recursive filter : y[n] = \sum_{k=0}^{N} b_k x[n-k] + \sum_{m=1}^{M} a_m y[n-m]

#### Linear Systems
