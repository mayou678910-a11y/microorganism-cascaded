Dataset
This project is benchmarked on the Environmental Microorganism Dataset Version 6 (EMDS-6), consisting of 21 taxonomic classes with 40 paired brightfield micrographs and binary ground-truth masks per class (840 image-mask pairs total). In the ground-truth masks, the target microorganism is white (255) and the background is black (0).

Classes (21):
01 Actinophrys, 02 Arcella, 03 Aspidisca, 04 Codosiga, 05 Colpoda, 06 Epistylis, 07 Euglypha, 08 Paramecium, 09 Rotifera, 10 Vorticella, 11 Noctiluca, 12 Ceratium, 13 Stentor, 14 Siprostomum, 15 Keratella Quadrala, 16 Euglena, 17 Gymnodinium, 18 Gonyaulax, 19 Phacus, 20 Stylongchia, 21 Synchaeta.

Preparation for Retraining:
To retrain the models from scratch, download EMDS5-Original.zip and EMDS5-Ground Truth.zip from this repository's Releases tab and extract them directly into the root directory alongside 41.ipynb (the extracted folders retain their benchmark legacy prefix EMDS5-Original and EMDS5-Ground Truth).

Acknowledgements:
Special thanks to Prof. Dr.-Ing. Chen Li and the research team for creating and releasing the EMDS-6 dataset.
