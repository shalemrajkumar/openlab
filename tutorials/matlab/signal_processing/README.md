## Signal Processing

- Data Aquisition
- Signal conditioning
- Feature extraction
- Hypothesis testing


### Data Aquisition

- Recording the signal with maximizing SNR
- Sampling the analog signal to digital signal. without losing information

### analog to digital conversion

context:

the signal is recored a continuous electronic (voltage) signal. the signal is sampled and quantized to a finite set of values.

- Sampling
    - The process of converting a continuous signal into a discrete signal by taking samples at regular intervals.
    - $ x[n] = x(nT_s) $
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
    
`convention`
    - T_s : sampling period
    - F_s : sampling frequency

- Sampling causes loss of information

- Example:
    - $$ x[n] = a \cos(2\pi f_0 n T_s + \phi) $$
    - what could be the original x(t) if sampled at T_s?
        - may be a $ x(t) = a\cos(2\pi f_0 n t + \phi) $, but not necessarily.
        - $ x(t) = a \cos(2\pi (f_0 + mF_s)t + \phi) $
        - there are infinitely many possible x(t) that could produce the same samples.
    

    - The Nyquist-Shannon sampling theorem states that a signal can be completely reconstructed from its samples if it is sampled at a rate greater than twice its highest frequency component.
    - $ f_s > 2f_{max} $
    - where $ f_{max} $ is the maximum frequency of the signal.
    - If the sampling rate is too low, aliasing occurs, causing high-frequency components to be misrepresented as lower frequencies.

