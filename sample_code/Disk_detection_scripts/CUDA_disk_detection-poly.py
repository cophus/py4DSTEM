import py4DSTEM
import numpy as np
import h5py
import time
from py4DSTEM.process.diskdetection.diskdetection_cuda import  find_Bragg_disks_CUDA

def main():
    
    #set the file path
    file_path_input = 'Ge_SiGe_ML_ideal.h5'
    
    # load the dataset and the probe
    dataset = py4DSTEM.io.read(file_path_input, data_id=0)
    probe = py4DSTEM.io.read(file_path_input, data_id='probe');

    # Estimate the radius of the BF disk, and the center coordinates
    probe_semiangle, qx0, qy0 = py4DSTEM.process.calibration.get_probe_size(
        probe.data)
    # Generate the probe kernel
    probe_kernel = py4DSTEM.process.diskdetection.get_probe_kernel_edge_sigmoid(
        probe.data, 
        probe_semiangle * 0.0,                                                        
        probe_semiangle * 2.0,)
    # generate Fourier Transform of the probe 
    probe_kernel_FT = np.conj(np.fft.fft2(probe_kernel))

    # set hyperparameters
    corrPower = 0.8
    sigma_gaussianFilter = 5
    edgeBoundary = 20
    maxNumPeaks = 100
    minPeakSpacing = 125
    minRelativeIntensity = 0.005
    
    start = time.time()
    
    quicker but less good method 
    CUDA_peaks = find_Bragg_disks_CUDA(
                dataset,
                probe_kernel_FT,
                corrPower=corrPower,
                sigma=sigma_gaussianFilter,
                edgeBoundary=edgeBoundary,
                minRelativeIntensity=minRelativeIntensity,
                minPeakSpacing=minPeakSpacing,
                maxNumPeaks=maxNumPeaks,
                subpixel='poly',
                )
    

    # # slower but better method
    # CUDA_peaks = find_Bragg_disks_CUDA(
    #             dataset,
    #             probe_kernel,
    #             corrPower=corrPower,
    #             sigma=sigma_gaussianFilter,
    #             edgeBoundary=edgeBoundary,
    #             minRelativeIntensity=minRelativeIntensity,
    #             minPeakSpacing=minPeakSpacing,
    #             maxNumPeaks=maxNumPeaks,
    #             subpixel='multicorr',
    #             )
    end = time.time()

    run_time = end - start
    print(run_time) 
    return run_time, CUDA_peaks


if __name__ == '__main__':
    main()