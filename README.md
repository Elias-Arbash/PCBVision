# PCB-Vision: A Multiscene RGB-Hyperspectral Benchmark Dataset of Printed Circuit Boards

## Overview

Our primary focus is to enhance the non-invasive optical analysis of E-waste materials, specifically plastics and printed circuit boards (PCBs). We aim to develop a smart multisensor network that utilizes RGB cameras and hyperspectral imaging, along with other types of sensors, to improve the efficiency of the E-waste recycling industry. This involves providing both quantitative and qualitative information that aids in decision-making for subsequent sorting and processing.

## Research Paper

This GitHub repository corresponds to a research paper titled "PCB-Vision: A Multiscene RGB-Hyperspectral Benchmark Dataset of Printed Circuit Boards." The paper introduces the first RGB-Hyperspectral (HS) benchmark segmentation dataset for PCBs. You can access the paper [here](https://arxiv.org/abs/2401.06528).

## Dataset Details

The dataset includes:
- RGB images of 53 PCBs scanned with a high-resolution RGB camera (Teledyne Dalsa C4020).
- 53 hyperspectral data cubes of those PCBs scanned with Specim FX10 in the VNIR range.
- Two segmentation ground truth files: 'General' and 'Monoseg' for 4 classes of interest - 'others,' 'IC,' 'Capacitor,' and 'Connectors.'

## Code Repository

The repository includes code to read, manipulate, and process large-scale data. It also provides examples of preprocessing and processing steps, such as dimensionality reduction (e.g., PCA) and image segmentation using 5 deep learning models.

### Requirements

To use the codes without errors, install the libraries listed in the Requirements.txt file. The codes require at least 1 GPU to run and handle the data.

### Usage

For detailed code instructions, please refer to the code documentations. More information about the methodology and experiments can be found in the paper [here](https://arxiv.org/abs/2401.06528).

## Data Access

To utilize the dataset, download it from [this link](https://rodare.hzdr.de/record/2704).

## Contributions

All comments and contributions are welcomed. The repository can be forked, edited, and pushed to different branches for enhancements. Feel free to contact me directly at e.arbash@hzdr.de or via our [website](https://www.iexplo.space/).

## License

The code is licensed under the Apache-2.0 license. Any further development and application using this work should be opened and shared with the community.

## Citation

When using the materials of the work and dataset, please cite it as follows:

**Latex:**
```latex
@article{arbash2024pcb,
  title={PCB-Vision: A Multiscene RGB-Hyperspectral Benchmark Dataset of Printed Circuit Boards},
  author={Arbash, Elias and Fuchs, Margret and Rasti, Behnood and Lorenz, Sandra and Ghamisi, Pedram and Gloaguen, Richard},
  journal={arXiv preprint arXiv:2401.06528},
  year={2024}
}
