# MQAUNet
This is the official code repository for "Beyond Convolution: A Morphology-Guided Quantum-Inspired Attention UNet for Medical Image Segmentation", 
**0. Main Environments**
- python 3.11.4
- pytorch 2.8.0

**1. Prepare the dataset.**

**The ISIC17, ISIC18, Kvasir-SEG and CVC-ClinicDB datasets, divided into a 8:1:1 ratio, can be found here:** { https://www.kaggle.com/datasets/johnchfr/isic-2017, https://www.kaggle.com/datasets/trantoanthang/isic-2018, https://www.kaggle.com/datasets/debeshjha1/kvasirseg, https://www.kaggle.com/datasets/balraj98/cvcclinicdb.}

- After downloading the datasets, you are supposed to put them into './data/isic2017/', './data/isic2018/', './data/ Kvasir-SEG /',  and './data/ CVC-ClinicDB /' and the file format reference is as follows. (take the ISIC17 dataset as an example.)
- './data/isic17/'
    - images
      - .png
    - masks
      - .png
 

**2. Train the MQAUNet.**
```
cd MQAUNet
```
```
python train.py
```

**3. Obtain the outputs.**
- After trianing, you could obtain the outputs in './results/'

