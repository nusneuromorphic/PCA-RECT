Event-based Recognition for FPGA implementation

The VLFeat Library is provided with the repo and the MATLAB script configures it on-the-fly.

Needs MATLAB AER Vision Functions from Garrick Orchard (https://github.com/gorchard/Matlab_AER_vision_functions)

There are three version of the code:

   FPGAModular: For evaluating the exact modular version where FPGA is closely followed
   
   FAST: 100x faster Quick floating-point versions for parameter testing (YH and RB)
   
   FASTnoPCA: Fast version without principal component analysis (YH and RB)
   
   FASTnoPCAwithDet: Fast version with detector incorporated (no PCA) (YH and RB)
   
At the moment, the training files are in our neuromorphic shared drive and needs to be placed one folder level above the working folder 
(the code uses '../' to reference the files). 

\Storage\Recognition_FPGA_trainfiles

-----Instruction for Tunning Parameters(testing)-----(Yang Hong)
Tune descriptor size: 1) set value of "param.descsize=7"  2) CTRL+H to replace "5by5" to "7by7"
Tune codebook size: 1) set value of "histopts.num_bins=150" 2) CTRL+H to replace "100codebok" to "150codebok"

Then, you can run the code to load or get your data properly.






