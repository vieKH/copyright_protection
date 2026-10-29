"""Blind geometric search in the Fourier magnitude domain.

IMPORTANT: this IS frequency-domain geometric synchronization. It does not
invert/rotate/resize the image and does not use the host, payload bits, or true
attack parameters. Random-phase embedding is unchanged. Extraction uses
magnitudes, so the phase map is not a decoder secret in this method.
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, binary_fill_holes
from .utils import qr_to_spectrum_positions


@dataclass(frozen=True)
class SpectralGeometryResult:
    bits: np.ndarray
    score: np.ndarray
    threshold: float
    estimate: float
    quality: float
    supported_bits: int
    hypotheses: int


def extract_spectral_geometry(image, payload_size, block_size, kind, *,
                              x=None, y=None, offset=None, gap=1,
                              scale_min=.5, scale_max=2., scale_step=.002,
                              angle_step=.25):
    """Search a fixed PUBLIC range, shared across all attack conditions.

    Rotation estimates are modulo 180 degrees: conjugate Fourier magnitudes
    carry the same payload. No true-bit/reference/actual-angle argument exists.
    Out-of-Nyquist slots are excluded from the template selection statistic;
    they are reported as unsupported and assigned zero at the binary output.
    They still count in the reported overall bit accuracy, without exclusion.
    """
    img = np.asarray(image)
    if img.ndim != 2 or min(img.shape) < block_size or not np.isfinite(img).all():
        raise ValueError('Expected a finite grayscale image containing at least one block')
    if kind not in ('scale','rotation'):
        raise ValueError('kind must be scale or rotation')
    if not (np.isfinite(scale_min) and np.isfinite(scale_max) and 0 < scale_min < scale_max):
        raise ValueError('Invalid fixed scale search range')
    if not np.isfinite(scale_step) or scale_step<=0 or not np.isfinite(angle_step) or angle_step<=0:
        raise ValueError('Search steps must be finite and positive')
    if int(np.ceil((scale_max-scale_min)/scale_step))>10000 or 180/angle_step>10000:
        raise ValueError('At most 10000 coarse hypotheses are allowed')
    height,width=img.shape
    # Center before zero padding to interpolate complex spectra without a large
    # artificial phase ramp; mean subtraction suppresses padded DC sidelobes.
    padded=np.zeros((2*height,2*width),dtype=np.float64)
    rr,cc=height//2,width//2
    # Remove observed zero-filled exterior before centering. Otherwise a
    # flat image rotated onto black padding creates a much stronger spectrum
    # than the watermark itself. No clean image or attack parameter is used.
    valid=binary_fill_holes(img!=0)
    if valid.mean()<.2:
        valid=np.ones(img.shape,dtype=bool)
    centered=np.where(valid,img.astype(float)-float(np.mean(img[valid])),0.)
    padded[rr:rr+height,cc:cc+width]=centered
    magnitude=np.fft.fftshift(np.abs(np.fft.fft2(np.fft.ifftshift(padded))))
    noise=gaussian_filter(magnitude,6.)+np.sqrt(height*width)*.1
    whitened=magnitude/noise
    pp=qr_to_spectrum_positions(payload_size,block_size,x,y,offset,gap=gap)
    freq=np.fft.fftfreq(block_size)*block_size
    # Signed cycles per pixel, independent of observed image/canvas dimensions.
    u=np.array([freq[p['row']] for p in pp])/block_size
    v=np.array([freq[p['col']] for p in pp])/block_size

    def evaluate(values):
        values=np.asarray(values,dtype=float)
        if kind=='scale':
            fu=u[None,:]/values[:,None]; fv=v[None,:]/values[:,None]
        else:
            theta=np.deg2rad(values)[:,None]
            fu=u[None,:]*np.cos(theta)-v[None,:]*np.sin(theta)
            fv=u[None,:]*np.sin(theta)+v[None,:]*np.cos(theta)
        valid=(abs(fu)<.495)&(abs(fv)<.495)
        samples=map_coordinates(whitened,np.array([2*height*fu+height,2*width*fv+width]),
                                order=1,mode='constant',cval=0.)
        samples=np.where(valid,samples,0.)
        count=valid.sum(axis=1)
        # Fixed template support, not the true positions of one-bits.
        quality=np.sum(np.log1p(samples),axis=1)/np.maximum(count,1)
        quality=np.where(count>=max(4,int(.2*payload_size**2)),quality,-np.inf)
        return quality,samples,valid

    if kind=='scale':
        coarse=np.linspace(scale_min,scale_max,int(np.ceil((scale_max-scale_min)/scale_step))+1)
    else:
        coarse=np.arange(-90.,90.,angle_step)
    quality,_,_=evaluate(coarse)
    if not np.isfinite(quality).any():
        raise ValueError('No search hypothesis has enough supported carriers')
    center=float(coarse[int(np.argmax(quality))])
    if kind=='scale':
        fine=np.linspace(max(scale_min,center-2*scale_step),min(scale_max,center+2*scale_step),41)
    else:
        fine=np.linspace(center-1.2*angle_step,center+1.2*angle_step,61)
        fine=(fine+90.)%180.-90.
    quality,scores,valid=evaluate(fine)
    index=int(np.argmax(quality)); score=scores[index]; support=valid[index]
    threshold=float(np.median(score[support]))
    bits=((score>threshold)&support).astype(np.uint8).reshape(payload_size,payload_size)
    return SpectralGeometryResult(bits,score.reshape(payload_size,payload_size),threshold,
                                  float(fine[index]),float(quality[index]),int(support.sum()),
                                  len(coarse)+len(fine))
