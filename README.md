# Classifying Zooplankton in Lake Huron
A machine learning project to classify zooplankton in Lake Huron using image and plain-text data. Data provided by Ontario Ministry of Natural Resources (MNR). 

The methods used in this project were informed by [Collaborative Deep Learning Models to Handle
Class Imbalance in FlowCam Plankton Imagery](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9187202) by Kerr et al. (2020). While the authors used RGB plankton imagery from the English Channel, the overall data structure is very similar to that provided by the MNR. Therefore, a **collaborative deep learning approach** was used to classify the Lake Huron zooplankton data. 



## Table of Contents
- [Data](#data)
- [Frameworks](#frameworks)
- [Installation](#installation)
- [Usage](#usage)
- [Credits](#credits)


## Data
Data from this project is provided by the MNR. There is both plain-text data and image data for each water sample. 

Due to memory limitations on Github, not all data necessary for the project could be uploaded to this repository. Instructions for manual download described in [Installation](#installation).

## Frameworks
This project was developed with Tensorflow, Keras, and Sci-kit Learn. 


## Installation
1. Clone the repository in your terminal:
```bash
 git clone https://github.com/alauzon13/ZooplanktonGit/
```
2. Download data from Onedrive: 

From [Onedrive](https://utoronto-my.sharepoint.com/:f:/g/personal/vianey_leosbarajas_utoronto_ca/ElpxgGCqDHtJjFml4UJnD_QBPH7a3ijH_NCV-btbCNvbbw?e=qG1M9a), download the HURON_OverlapTiffsWithPP folder, including the HURONOverlap_csv sub-folder. Access may be required. Email adele.lauzon@mail.utoronto.ca with any questions about data access.

## Usage   
The full project pipeline is outlined in the notebook FullPipeline.ipynb. This is what was used to train and test the full models. Because the models take up to 3 hours to converge, a Mini Example was also created to provide an easier way to interact with the project. The notebook and necessary files for this are in MiniExample. 

## Credits 

This project was completed with support from professor [Dr. Vianey Leos Barajas](https://www.statistics.utoronto.ca/people/directories/all-faculty/vianey-leos-barajas) and post-doctoral fellow [Dr. Sofia Ruiz Suarez](https://www.statistics.utoronto.ca/people/directories/postdoctoral-fellows/sofia-ruiz-suarez). 




