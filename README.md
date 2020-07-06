# PCA-RECT: Event-based Object Detection and Classification

The VLFeat Library is provided with the repo and the MATLAB script configures it on-the-fly.

Needs [MATLAB AER Vision Functions](https://github.com/gorchard/Matlab_AER_vision_functions) from Garrick Orchard.

There are four versions of the code:

   FPGAModular: For evaluating the exact modular version where FPGA is closely followed.
   
   FAST: 100x faster Quick floating-point versions for parameter testing.
   
   FASTnoPCA: Fast version without principal component analysis.
   
   FASTnoPCAwithDet: Fast version with detector incorporated (no PCA).
   
The training files can be found in the [N-SOD Dataset](https://tinyurl.com/s84nlm4) and needs to be placed in the correct path, relative to the main executing file. (the code uses '../' to reference the files). 

../N-SOD Datatet/


## Instructions to execute

   1. Download the N-SOD Dataset and place above the PCA-RECT folder.
   2. Add to path the MATLAB AER Vision Functions.
   3. Run one of the scripts, e.g. Event_context_DEMOuav_rmax7by7rect_FAST

## Instructions for Tunning Parameters (testing)
Tune descriptor size: 
   1. Set value of "param.descsize=7"
   2. CTRL+H to replace "5by5" to "7by7"
   
Tune codebook size: 
   1. Set value of "histopts.num_bins=150"
   2. CTRL+H to replace "100codebok" to "150codebok"

Then, you can run the code to load or get your data properly.

## Citations ##
Ramesh B., Ussa A., Vedova L.D., Yang H., Orchard G. (2020) Low-Power Dynamic Object Detection and Classification With Freely Moving Event Cameras. Front. Neurosci. 14:135 doi: 10.3389/fnins.2020.00135

```bibtex
@ARTICLE{10.3389/fnins.2020.00135,
   AUTHOR="Ramesh, Bharath and Ussa, Andrés and Della Vedova, Luca and Yang, Hong and Orchard, Garrick",
   TITLE="Low-Power Dynamic Object Detection and Classification With Freely Moving Event Cameras",
   JOURNAL="Frontiers in Neuroscience",
   VOLUME="14",
   PAGES="135",
   YEAR="2020",
   DOI="10.3389/fnins.2020.00135",
   ISSN="1662-453X"
}
```

Ramesh B., Ussa A., Vedova L.D., Yang H., Orchard G. (2019) PCA-RECT: An Energy-Efficient Object Detection Approach for Event Cameras. In: Carneiro G., You S. (eds) Computer Vision – ACCV 2018 Workshops. ACCV 2018. Lecture Notes in Computer Science, vol 11367. Springer, Cham

```bibtex
@InProceedings{10.1007/978-3-030-21074-8_35,
   author="Ramesh, Bharath and Ussa, Andr{\'e}s and Vedova, Luca Della and Yang, Hong and Orchard, Garrick",
   editor="Carneiro, Gustavo and You, Shaodi",
   title="PCA-RECT: An Energy-Efficient Object Detection Approach for Event Cameras",
   booktitle="Computer Vision -- ACCV 2018 Workshops",
   year="2019",
   publisher="Springer International Publishing",
   address="Cham",
   pages="434--449",
   isbn="978-3-030-21074-8"
}
```
