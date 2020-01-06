# PCA-RECT: Event-based Object Detection and Classification

The VLFeat Library is provided with the repo and the MATLAB script configures it on-the-fly.

Needs [MATLAB AER Vision Functions](https://github.com/gorchard/Matlab_AER_vision_functions) from Garrick Orchard.

There are four versions of the code:

   FPGAModular: For evaluating the exact modular version where FPGA is closely followed
   
   FAST: 100x faster Quick floating-point versions for parameter testing (YH and RB)
   
   FASTnoPCA: Fast version without principal component analysis (YH and RB)
   
   FASTnoPCAwithDet: Fast version with detector incorporated (no PCA) (YH and RB)
   
The training files can be found in the [N-SOD Dataset](https://tinyurl.com/s84nlm4) and needs to be placed in the correct path, relative to the main executing file. (the code uses '../' to reference the files). 

../../N-SOD Datatet/

## Instruction for Tunning Parameters (testing)
Tune descriptor size: 
   1. Set value of "param.descsize=7"
   2. CTRL+H to replace "5by5" to "7by7"
   
Tune codebook size: 
   1. Set value of "histopts.num_bins=150"
   2. CTRL+H to replace "100codebok" to "150codebok"

Then, you can run the code to load or get your data properly.
