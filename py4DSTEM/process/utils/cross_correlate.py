# Cross correlation function

import numpy as np
import matplotlib.pyplot as plt


def get_cross_correlation(
    ar,
    template,
    corrPower=1,
    _returnval='real'
    ):
    """
    Get the cross/phase/hybrid correlation of `ar` with `template`, where
    the latter is in real space.

    If _returnval is 'real', returns the real-valued cross-correlation.
    Otherwise, returns the complex valued result.
    """
    assert( _returnval in ('real','fourier') )
    template_FT = np.conj(np.fft.fft2(template))
    return get_cross_correlation_FT(
        ar,
        template_FT,
        corrPower = corrPower,
        _returnval = _returnval)


def get_cross_correlation_FT(
    ar,
    template_FT,
    corrPower = 1,
    _returnval = 'real',
    mask = None,
    ):

    """
    Get the cross/phase/hybrid correlation of `ar` with `template_FT`, where
    the latter is already in Fourier space (i.e. `template_FT` is
    `np.conj(np.fft.fft2(template))`.

    If _returnval is 'real', returns the real-valued cross-correlation.
    Otherwise, returns the complex valued result.
    """

    assert(_returnval in ('real','fourier'))

    if mask is None:
        # unnormalized correlation
        if corrPower == 1:
            cc = np.fft.fft2(ar) * template_FT
        else:
            m = np.fft.fft2(ar) * template_FT
            cc = np.abs(m)**(corrPower) * np.exp(1j*np.angle(m))
    else:
        # normalized correlation
        if corrPower == 1:
            # cc = np.real(np.fft.ifft2(np.fft.fft2(ar * mask) * template_FT)) 

            ar_fft = np.fft.fft2(ar)
            mask_fft = np.fft.fft2(mask)
            template = np.fft.ifft2(template_FT)

            cc = np.real(np.fft.ifft2(np.fft.fft2(ar * mask) * template_FT)) 

            # normalize the correlation
            # cc_norm = 
            # cc_norm = (mask - 1) * 
            term_0 = np.real(np.fft.ifft2(mask_fft * template_FT))
            cc_norm_num = ar * term_0
            cc_norm_den = 1 * \
                ( np.fft.ifft2(np.fft.fft2(template**2) * mask_fft) * mask - term_0)

            # cc_norm = np.real(np.fft.ifft2(np.fft.fft2(mask) * template_FT))
            # # sub = np.logical_and(cc_norm > 0.01*np.max(np.abs(cc_norm)), cc > 0)
            # # cc[sub] /= cc_norm[sub]
            # # cc[np.logical_not(sub)] = 0
            # thresh = np.max(cc_norm) * 0.001
            # # sub = cc_norm
            
            fig,ax = plt.subplots(figsize=(20,7))
            ax.imshow(
                # np.real(cc_norm_den),
                np.hstack((cc * mask, cc_norm_num)),
                # cc / cc_norm,
                vmin = 0,
                vmax = 4,
                cmap='turbo',
                )

            cc = np.fft.fft2(cc)


    if _returnval == 'real':
        cc = np.maximum(np.real(np.fft.ifft2(cc)),0)
    return cc



def get_shift(
    ar1,
    ar2,
    corrPower=1
    ):
    """
	Determine the relative shift between a pair of arrays giving the best overlap.

	Shift determination uses the brightest pixel in the cross correlation, and is
    thus limited to pixel resolution. corrPower specifies the cross correlation
    power, with 1 corresponding to a cross correlation and 0 a phase correlation.

	Args:
		ar1,ar2 (2D ndarrays):
        corrPower (float between 0 and 1, inclusive): 1=cross correlation, 0=phase
            correlation

    Returns:
		(2-tuple): (shiftx,shifty) - the relative image shift, in pixels
    """
    cc = get_cross_correlation(ar1, ar2, corrPower)
    xshift, yshift = np.unravel_index(np.argmax(cc), ar1.shape)
    return xshift, yshift



