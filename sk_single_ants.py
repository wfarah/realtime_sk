import numpy as np
import matplotlib.pyplot as plt

NANTS     = 20
NFREQS    = 192
NTIME     = 8192
NPOLS     = 2

# Number of blocks to estimate 
# random number variables to replace
# flagged data
N_EST     = 16

# These are theoretical values
# Obtained from: 
# https://www.worldscientific.com/doi/10.1142/S225117171940004X
M         = NTIME
SK_MEAN   = 1
SK_STD    = np.sqrt(4. / M)

# User input, needs to be obtained from the backend configuration
SK_THRESH = 6  

def get_new_block():
    """
    Assuming I am receiveing data in the form of: 
    AFTP (slow -> fast axis)
    Antenna -> Frequency -> Time -> Polarization

    I'm simulating data with mean = 0, std = 10, which is
    roughly what we expect to see. NOT checking for overflow, 
    but doesn't matter for simulations here 
    (might matter quite a bit though for real data though!)
    """
    rl = np.round(np.random.normal(0, 10,
         size = (NANTS, NFREQS, NTIME, NPOLS)))
    im = np.round(np.random.normal(0, 10,
         size = (NANTS, NFREQS, NTIME, NPOLS)))

    block = rl + 1j*im
    return block.astype(np.complex64)

def get_mad(block, median):
    """
    get median absolute deviation
    https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    mad = np.median(np.abs(block - median))
    return mad


"""
def get_new_block():
    # return real data from a guppi file
    raise NotImplementedError
"""

sk_vals = np.zeros((NANTS, NFREQS, NPOLS), dtype=float)
sk_mask = np.zeros((NANTS, NFREQS, NPOLS), dtype=bool)
acc_some_blocks = []

some_observation_blocks = 500
for i in range(some_observation_blocks):
    print(i)
    # Obtain a new block
    block = get_new_block()

    # Let's estimate the mean and std of the incomming data
    # This is one way to do so, but we can think of other ways
    if i < N_EST:
        acc_some_blocks.append(block)
        continue
    if i == N_EST:
        print("Estimating statistics for replacing data...")
        rl = np.real(acc_some_blocks)
        med_rl = np.median(rl)
        mad_rl = get_mad(rl, med_rl)
        std_rl = 1.4826*mad_rl

        im = np.real(acc_some_blocks)
        med_im = np.median(im)
        mad_im = get_mad(im, med_im)
        std_im = 1.4826*mad_im
        print("Done")

    # Start calculating the SK
    for iant in range(NANTS):
        for ifreq in range(NFREQS):
            for ipol in range(NPOLS):
                # get the data for that freq channel
                d = block[iant, ifreq, :, ipol]
                # detect data
                d_dt = d.real * d.real + d.imag * d.imag

                # Calculate statistics
                s1 = d_dt.sum()
                s2 = (d_dt**2).sum()

                # The Spectral Kurtosis
                sk = (M + 1.)/(M - 1) * (M*s2/s1**2 - 1)
                sk_vals[iant, ifreq, ipol] = sk

                sk_bool = (sk < SK_MEAN - SK_STD * SK_THRESH) or\
                          (sk > SK_MEAN + SK_STD * SK_THRESH)

                # Now add the boolean flag
                # We can write this array on disk for later processing
                sk_mask[iant, ifreq, ipol] = sk_bool

                # Now replace data if RFI is detected
                if sk_bool:
                    # The below would be different in the real world
                    # given data rounding and casting back to 8bit
                    sim = np.round(np.random.normal(mad_rl, mad_rl,
                            size = NTIME)) +\
                          np.round(np.random.normal(mad_im, mad_im,
                            size = NTIME)) * 1j
                    block[iant, ifreq, :, ipol] = sim
