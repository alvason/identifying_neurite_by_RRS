# Identifying neurite by RRS method
This repository contains code for implementing the RRS method initially described in the paper:

<https://www.nature.com/articles/s41598-019-39962-0)>

```
Random-Reaction-Seed Method for Automated Identification of Neurite Elongation and Branching
by Alvason Z. Li, Lawrance Corey, and Jia Zhu, (2019)
```
(is still working on this repository, a AlvaHmm package will be ready soon...)
## Overview
### tracing neurite in microfuidic device
![](https://github.com/alvason/identifying_neurite_by_RRS/blob/master/figure/AlvaHmm_demo_edge_detection_selected_seeding_selected_seed_window0.jpg)
![](https://github.com/alvason/identifying_neurite_by_RRS/blob/master/figure/AlvaHmm_demo_edge_detection_selected_seeding_connected_way_window3.png)


## Installation
### Prerequisites
This code is written and tested in Python 3.6.5.
The required Python libaries are:
* NumPy
* SciPy
* Matplotlib
* AlvaHmm

### Getting started
```
clone git https://github.com/alvason/identifying_neurite_by_RRS.git
```
### Examples
#### [Demo 1 of RRS code in Jupyter-notebook](https://github.com/alvason/identifying_neurite_by_RRS/blob/master/code/AlvaHmm_demo_seeding_map/AlvaHmm_demo_random_reaction_seed_by_blob_map.ipynb)
#### [Demo 2 of RRS code in Jupyter-notebook](https://github.com/alvason/identifying_neurite_by_RRS/blob/master/code/AlvaHmm_demo_seeding_map/AlvaHmm_demo_random_reaction_seed_by_random_map.ipynb)
![](https://github.com/alvason/identifying_neurite_by_RRS/blob/master/code/AlvaHmm_demo_seeding_map/figure/AlvaHmm_demo_random_reaction_seed_by_blob_map.png)
![](https://github.com/alvason/identifying_neurite_by_RRS/blob/master/code/AlvaHmm_demo_seeding_map/figure/AlvaHmm_random_map_vs_blob_map.png)
## Contact and more information
This repository is maintained by [@alvason](https://github.com/alvason).
