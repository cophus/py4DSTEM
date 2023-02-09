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
            f1 = ar.copy()
            f2 = np.real(np.fft.ifft2(np.conj(template_FT)))
            f2p = np.rot90(f2, 2)
            F1 = np.fft.fft2(f1)
            F2s = template_FT.copy()
            M1 = np.fft.fft2(mask)
            M2s = np.conj(np.fft.fft2(np.ones(mask.shape)))

            cc = np.real(np.fft.ifft2(F1 * F2s))

            dm = np.real(np.fft.ifft2(M1 * M2s))

            tn0 = cc
            tn1 = np.real(np.fft.ifft2(F1 * M2s)) * np.real(np.fft.ifft2(M1 * F2s)) / dm


            td0 = np.sqrt(np.maximum(
                np.real(np.fft.ifft2(np.fft.fft2(f1**2) * M2s)) \
                - (np.real(np.fft.ifft2((F1 * M2s))))**2 / dm,
                0)) 
            # td1 = np.sqrt(np.maximum(
            #     np.real(np.fft.ifft2(M1 * np.fft.fft2(f2p))) \
            #     - np.real(np.fft.ifft2(M1*F2s))**2 / dm,
            #     0))
            td1 = np.sqrt(np.maximum(
                np.real(np.fft.ifft2(M1 * np.fft.fft2(f2p**2))) \
                - np.real(np.fft.ifft2(M1*F2s))**2 / dm,
                0))
            td = td0 * td1
            # print(np.median(td), np.sqrt(np.mean(f1**2)), np.sqrt(np.mean(f2**2)))


            cc_norm = tn0 - tn1
            sub = td > 0
            # cc_norm[sub] /= td[sub]
            cc_norm *= mask

            sub = tn1 > 0.5*np.max(tn1)
            print(np.sqrt(np.median(cc[sub]**2)), np.sqrt(np.median(tn1[sub]**2)))

            td = td0 * td1

            fig,ax = plt.subplots(figsize=(16,5))
            h = ax.imshow(
                # td0 * td1,
                np.hstack((ar/10,cc, cc_norm)),
                # mask/(cc + 1e-3),
                vmin = 0,
                vmax = 1,
                # cmap='turbo',
                )
            bar = plt.colorbar(h)
            # print(
            #     np.sqrt(np.mean(tn0**2)),
            #     np.sqrt(np.mean(tn1**2)),
            #     np.sqrt(np.mean(td0**2)),
            #     np.sqrt(np.mean(td1**2)),
            # )

            # cc_norm = (np.real(np.fft.ifft2(F1 * F2s)) \
            #     - 0) / \


            # ar_mean = np.mean(ar)
            # ar2_mean = np.mean(ar**2)
            # mask_mean = np.mean(mask)

            # # ar_fft = np.fft.fft2(ar)
            # mask_fft = np.fft.fft2(mask)
            # template = np.fft.ifft2(template_FT)

            # cc = np.real(np.fft.ifft2(np.fft.fft2(ar) * np.conj(template_FT)))

            # term_0 = np.real(np.fft.ifft2(mask_fft * np.conj(template_FT)))
            # cc_norm_num = term_0  * (ar_mean / mask_mean)

            # t1 = np.real(np.fft.ifft2(mask_fft*np.fft.fft2(template**2)))
            # t2 = (term_0**2) / (mask_mean * mask.size)
            # cc_norm_den = np.sqrt(np.maximum(t1-t2,0))
            # cc_norm_den *= np.sqrt(np.maximum(ar2_mean - ar_mean**2 / (mask_mean * mask.size),0))

            # # cc_norm_den = (1/mask_mean) * np.sqrt(mask_mean*ar2_mean - ar_mean**2) * \
            # #     np.sqrt(mask_mean*np.real(np.fft.ifft2(mask_fft * np.fft.fft2(template**0))) - term_0**2)
            # # cc_norm_den = (1/mask_mean) * np.sqrt(mask_mean*ar2_mean - ar_mean**2) * np.sqrt(np.clip(
            # #     mask_mean*np.real(np.fft.ifft2(mask_fft * np.fft.fft2(template**2))) - term_0**2, 0, np.inf))

            # # cc_norm_den = np.sqrt(np.clip(np.real(np.fft.ifft2(mask_fft*np.fft.fft2(template**2))) \
            # #     - (term_0**2) / (mask_mean*mask.size),0,np.inf))

            # # cc_norm_den *= (ar_mean / mask_mean)
            # # cc_norm_den *= ar2_mean

            # cc_norm = (cc - cc_norm_num) * mask
            # sub = cc_norm_den > 0
            # cc_norm[sub] /= cc_norm_den[sub]

            # scale = np.median(cc_norm[sub] / cc[sub])
            # print(scale, ar_mean)

            # print(np.min(cc_norm_den),np.max(cc_norm_den))

            # print(np.min(cc_norm_den), np.max(cc_norm_den))

            # normalize the correlation
            # cc_norm = 
            # cc_norm = (mask - 1) * 
            # term_0 = np.fft.ifft2(mask_fft * np.conj(template_FT))
            # cc_norm_num = ar * term_0
            # # cc_norm_den = ( np.fft.ifft2(np.fft.fft2(template**2) * mask_fft) * mask - term_0)
            # cc_norm_den = ar * np.sqrt(term_0**2 - \
            #     mask * np.fft.ifft2(mask_fft*np.fft.fft2(template**2))) * np.sqrt(1 - mask)
            # cc_norm_den = np.sqrt(1 - mask)
            # sub = cc_norm_den > 0
            # print(np.min(cc_norm_den))
            # cc = np.real(np.fft.ifft2(np.fft.fft2(ar * mask) * template_FT)) 

            # cc_norm = np.real(np.fft.ifft2(np.fft.fft2(mask) * template_FT))
            # # sub = np.logical_and(cc_norm > 0.01*np.max(np.abs(cc_norm)), cc > 0)
            # # cc[sub] /= cc_norm[sub]
            # # cc[np.logical_not(sub)] = 0
            # thresh = np.max(cc_norm) * 0.001
            # # sub = cc_norm
            
            # fig,ax = plt.subplots(figsize=(16,5))
            # ax.imshow(
            #     # cc,
            #     # cc_norm_den,
            #     # np.hstack((cc, (cc - cc_norm_num) / np.clip(cc_norm_den,1e-3,np.inf))),
            #     # np.hstack((cc, cc_norm/4)),
            #     # np.hstack((cc_norm_den, cc_norm_den_2)),
            #     # cc_norm_den,
            #     np.hstack((tn0, tn1, td0)),
            #     # 1/np.clip(cc_norm_den,1e-3,np.inf),
            #     # cc / cc_norm,
            #     vmin = 0,
            #     vmax = 10,
            #     cmap='turbo',
            #     )

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



