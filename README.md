Introduction of Environmental Microorganism Dataset Version 6 (EMDS-6)
	

In EMDS-6, there are 21 classes of environmental microorganisms (EMs).
In each calss, there are 40 EM original images and their corresponding binary groud truth images. 
	
In ground truth images, the foreground is white and background is black.
	

First release: 04-12-2021.
Nearst update: 04-12-2021.

Any questions: Prof. Dr.-Ing. Chen Li, lichen@bmie.neu.edu.cn 
Related people: Chen Li, Xuemin Zhu, Bolin Lu, Jinghua Zhang, Fangshu Ma, Yanling Zou, Peng Zhao, Pingli Ma, Hao Xu.
	

EMs:
	

class 01 Actinophrys
	

class 02 Arcella
	

class 03 Aspidisca
	

class 04 Codosiga
	

class 05 Colpoda
	

class 06 Epistylis
	

class 07 Euglypha
	

class 08 Paramecium
	

class 09 Rotifera
	

class 10 Vorticella
	

class 11 Noctiluca
	

class 12 Ceratium
	

class 13 Stentor
	

class 14 Siprostomum
	

class 15 Keratella Quadrala
	

class 16 Euglena
	

class 17 Gymnodinium
	

class 18 Gonyaulax
	

class 19 Phacus
	

class 20 Stylongchia
	

class 21 Synchaeta

## Dataset
The project is trained and benchmarked on the **EMDS-6** dataset (840 paired original micrographs and ground-truth masks across 21 classes). 

To retrain the models:
1. Download the dataset from the official release / publication [1].
2. Extract `EMDS5-Original` and `EMDS5-Ground Truth` into the project root directory.
3. Run the training cells in `41.ipynb`.
