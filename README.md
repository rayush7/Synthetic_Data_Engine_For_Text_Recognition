 # Synthetic_Data_Engine_For_Text_Recognition 

This repository contains code for Max Jaderberg's Text Renderer (https://arxiv.org/pdf/1406.2227.pdf) for generating synthetic cropped text images which can be used to train text recognition models like CRNN (https://arxiv.org/pdf/1507.05717.pdf) (https://github.com/bgshih/crnn)

### Dependencies(Successfully tested on Ubuntu 14.04 LTS)
* Anaconda Python 2.7
  https://www.continuum.io/downloads
  
* Opencv 3.1
   conda install -c menpo opencv3=3.1.0

* Pygame
  conda install -c cogsci pygame 
  
* LMDB  
  conda install -c conda-forge python-lmdb 
  
### To Run the Code :
* Change the paths in these files  
  * Synthetic_Data_Engine_For_Text_Recognition/SVT/font_path_list.txt
  * Synthetic_Data_Engine_For_Text_Recognition/SVT/icdar_2003_train.txt
  * Synthetic_Data_Engine_For_Text_Recognition/SVT/svt.txt
  * Synthetic_Data_Engine_For_Text_Recognition/text_renderer/generate_word_training_data.py
* cd Synthetic_Data_Engine_For_Text_Recognition/text_renderer/
* python generate_word_training_data.py (This code would generate the images and the lmdb files for training text recognition models like CRNN)
