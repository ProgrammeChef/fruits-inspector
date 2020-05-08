# Fruits Inspector – Computer Vision system for the visual inspection of fruits
<p align="center">
  <img src="https://github.com/bobcorn/fruits-inspector/blob/master/demo/gifs/thumb.gif">
</p>

Computer Vision system that is able to detect and locate defects and imperfections on fruits.

## Image characteristics
Fruits appearing in the images have been acquired through a NIR (Near Infra-Red) and a color camera with little parallax effect.

### First task
1. Images show three apples with clear external defects.

### Second task
1. Images show two apples with an unwanted reddish-brown area.

### Final challenge
1. Images show five kiwis, one of which with a clear external defect.

## Functional specifications
### First task
For each fruit appearing in each image, the vision system must provide the following information:

1. Outline the fruit by generating a binary mask.
2. Search for the defects on each fruit.

### Second task
For each fruit appearing in each image, the vision system must provide the following information:

1. Identify the russet or at least some part of it with no false positive areas (if possible), in order to correctly classify the two fruits.

### Final challenge
For each fruit appearing in each image, the vision system must provide the following information:

1. Segment the fruits and locate the defect in image “000007”. Special care should be taken to remove as “background” the dirt on the conveyor as well as the sticker in image “000006”.

## Performances
Performances are calculated as the average observed FPS of 10 000 consecutive software executions on a Intel Core i5 Dual-Core 2,7 GHz processor.

### First task
* 36 FPS

### Second task
* **Method 1 (K-means clustering)**: 0.4 FPS

* **Method 2 (Mahalanobis distance)**: 0.7 FPS

### Final challenge
* 40 FPS

## Full demo

<p align="center">
  <img src="https://github.com/bobcorn/fruits-inspector/blob/master/demo/gifs/full.gif">
</p>

## Requirements
The following Python packages must be installed in order to run the software:

* numpy
* opencv-python
* scipy
* scikit-learn

## Usage
Simply run the "main.py" script from terminal, after making sure it is located in the same directory of the "images" folder:

```bash
python main.py
```

or:

```bash
python3 main.py
```
