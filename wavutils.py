
import scipy
import scipy.io.wavfile

def singlechannel_stft(wavfile, framesz, hop):
    """
    Load a WAV file and return the STFT
    Fails if the WAV file is multi-channel
    framesz and hop are in seconds
    """
    sr, x = scipy.io.wavfile.read(wavfile)
    assert len(x.shape) == 1 # single-channel
    return stft(x, sr, framesz, hop)

def multichannel_stft(wavfile, framesz, hop, chan):
    """
    Load a WAV file and return the STFT of the selected channel
    framesz and hop are in seconds
    """
    sr, x = scipy.io.wavfile.read(wavfile)
    return stft(x[:,chan], sr, framesz, hop)

def autochannel_stft(wavfile, framesz, hop):
    """
    Load a WAV file and return the STFT
    If the WAV file is multi-channel, use the first channel
    framesz and hop are in seconds
    """
    sr, x = scipy.io.wavfile.read(wavfile)
    if len(x.shape) == 1:
        return stft(x, sr, framesz, hop)
    else:
        return stft(x[:,0], sr, framesz, hop)

# stft and istft from:
# http://stackoverflow.com/questions/2459295/invertible-stft-and-istft-in-python
def stft(x, fs, framesz, hop):
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hamming(framesamp)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp])
                     for i in range(0, len(x)-framesamp, hopsamp)])
    return X

def istft(X, fs, T, hop):
    x = scipy.zeros(T*fs)
    framesamp = X.shape[1]
    hopsamp = int(hop*fs)
    for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
        x[i:i+framesamp] += scipy.real(scipy.ifft(X[n]))
    return x

