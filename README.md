# Interactive Segmentation Benchmark

## What is it

This repo was made for comparing different interactive segmentation models. Current implementation includes:

- RITM
- Segment Anything Model (SAM)
- ClickSEG
- FastSAM
- SimpleClick

It was tested on grape bunch images. 
These images were imported from WGISD () and from Tannat pictures collected in Uruguay.

From WGISD were collected _Syrah_ and _Cabernet Suavignon_ images due it's similarities to the _Tannat_ variety. 

Images from the _Tannat_ and its ground truth data will be publicly available once our project get completed.
Right now they're under retention and were removed from this dataset.

## Models

This repo is ready for simulationg clicks on RITM, ClickSEG, FastSAM, SAM and SimpleClick after crops made by bounding boxes annotations.

### SAM
SAM has demonstrated poor performance when working that way.
So,  SAM is implemented to also accept inference on the whole image, using bounding boxes as prompts.
Additionally, it accepts a click as input. 
Altought SAM can work with more points associated with a single bounding box, in its repo there is an implementation limitation. 

### SimpleClick

The only way I found to run simpleclick inference was by resizing each image sent (croped from the dataset's bounding boxes) with fixed  input points. 

## Metodology

Two different approaches were made:

- For WGISD dataset, images already had instance masks and bounding boxes. _Fulano_ has annotated X, Y positions of each grape berry. Applying K-means with K=1 and k=3, click inputs were emulated with positive clicks. Instance masks were converted to semantic segmentation masks.
- For the Tannat dataset (available soon), using Supervisely platform were annotated semantic masks, bounding boxes and 3 points: the first one would always be positive. The two auxiliary could be positive or negative.

With two cases: 1 point (click) or 3 points, this data was used to simulate inputs with the intent of understanding the robustness of each model. 