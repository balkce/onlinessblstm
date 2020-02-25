#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
import numpy as np
import cmath
import soundfile as sf


def delay_f(x, time, fs):
    N = len(x)
    x_f = np.fft.fft(x)
    y_f = np.zeros(N, dtype=np.complex_)
    w = np.array(range(0, N / 2 + 1) + range(-N / 2 + 1, 0)) / float(N) * float(fs)
    for f in range(0, N):
        delay_factor = cmath.exp(-1j * 2 * math.pi * w[f] * time)  # steering vector for this frequency
        y_f[f] = x_f[f] * delay_factor
    y = np.fft.ifft(y_f).real
    return y


class BeamFormer:
    def __init__(self, d, m):
        self.d = d  # distance between microphones in meters
        self.c = 343.0  # speed of sound
        self.M = m  # number of microphones (should be 2)

    def phase_mask(
            self,
            X,
            doa_steer,
            phase_diff_threshold,
            N,
            nframes,
            fs,
    ):
        fft_win = 4 * nframes

        hann = np.hanning(fft_win)

        in_buff = np.zeros([self.M + 2, fft_win])
        out_buff = np.zeros([5, 3, fft_win])

        w = np.array(range(0, fft_win / 2 + 1) + range(-fft_win / 2 + 1, 0)) / float(fft_win) * float(fs)

        w_c = np.ones([self.M, fft_win], dtype=np.complex_)

        w_c[1, :] = np.exp(1j * 2 * math.pi * w * (self.d / self.c) * np.sin(doa_steer))

        X_f = np.zeros([self.M, fft_win], dtype=np.complex_)

        out_buff_ini_shift = int(fft_win * 3 / 4) - int(nframes / 2)
        out_buff_last_shift = int(fft_win / 4) - int(nframes / 2)

        o_ = np.zeros([5, N])

        for sample_i in range(0, N, nframes):
            in_buff[:, fft_win - nframes:fft_win] = X[:, sample_i:sample_i + nframes]

            in_buff_hann = np.multiply(in_buff, hann)

            X_f[0, :] = np.fft.fft(in_buff_hann[0])
            X_f[1, :] = np.fft.fft(in_buff_hann[1])

            X_f[1, :] = np.multiply(w_c[1, :], X_f[1, :])

            this_m0_phase = np.angle(X_f[0])
            this_m1_phase = np.angle(X_f[1])

            phase_diff = np.abs(this_m0_phase - this_m1_phase)

            freq_mask = np.array([phase_diff < phase_diff_threshold, phase_diff >= phase_diff_threshold]).astype(float)

            m0_signal_source = np.fft.ifft(X_f[0, :] * freq_mask[0]).real
            m1_signal_source = np.fft.ifft(X_f[1, :] * freq_mask[0]).real

            m0_signal_int = np.fft.ifft(X_f[0, :] * freq_mask[1]).real
            m1_signal_int = np.fft.ifft(X_f[1, :] * freq_mask[1]).real

            out_buff[0, 2, :] = (m0_signal_source + m1_signal_source) / 2.0  # np.fft.ifft(X_f[0, :] * freq_mask[0]).real
            out_buff[1, 2, :] = (m0_signal_int + m1_signal_int) / 2.0  # np.fft.ifft(X_f[0, :] * freq_mask[1]).real

            out_buff[2, 2, :] = in_buff_hann[0]  # m0
            out_buff[3, 2, :] = in_buff_hann[2]  # source
            out_buff[4, 2, :] = in_buff_hann[3]  # interf

            o_[:, sample_i:sample_i + nframes] = out_buff[:, 0,
                                                 out_buff_ini_shift:out_buff_ini_shift + nframes] + out_buff[:, 2,
                                                                                                    out_buff_last_shift:out_buff_last_shift + nframes]
            in_buff = np.roll(in_buff, -nframes, axis=1)
            out_buff = np.roll(out_buff, -1, axis=1)
        return o_


def main():
    d = 0.21  # distance between microphones in meters
    M = 2  # number of microphones (should be 2)
    c = 343.0

    doa1 = 20 * math.pi / 180.0
    doa2 = -40 * math.pi / 180.0
    doa3 = 80 * math.pi / 180.0
    doa_steer = doa1

    phase_diff_threshold = 30.0 * math.pi / 180.0

    (s1, samplerate1) = sf.read('25-88353-0001.flac')
    (s2, samplerate2) = sf.read('153-126652-0004.flac')
    (s3, samplerate3) = sf.read('392-131210-0003.flac')
    fs = samplerate1

    N = min(len(s1), len(s2), len(s3))
    nframes = 1024
    win_num = int(N / nframes)
    N = nframes * win_num
    N = nframes * (8 + 4 + 3)
    s1 = s1[0:N]
    s2 = s2[0:N]
    s3 = s3[0:N]

    X = np.zeros([M + 2, N])
    X[0, :] = s1 + s2 + s3
    X[1, :] = delay_f(np.array(s1), d / c * math.sin(doa1), fs) \
              + delay_f(np.array(s2), d / c * math.sin(doa2), fs) \
              + delay_f(np.array(s3), d / c * math.sin(doa3), fs)
    X[2, :] = s1
    X[3, :] = s2 + s3

    bf = BeamFormer(d=d, m=M)

    o_ = bf.phase_mask(X=X, doa_steer=doa_steer, phase_diff_threshold=phase_diff_threshold, N=N, nframes=nframes, fs=fs)
    o_ = o_[:, nframes * 4:nframes * -3]
    print(o_.shape)
    sf.write('phase_soi.wav', o_[0, :], fs)
    sf.write('phase_int.wav', o_[1, :], fs)

    sf.write('phase_m0.wav', o_[2, :], fs)
    sf.write('phase_original_source.wav', o_[3, :], fs)
    sf.write('phase_original_interf.wav', o_[4, :], fs)


if __name__ == "__main__":
    main()