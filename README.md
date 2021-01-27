# Identifying animal species in camera trap footage

## About the project

Using ResNet-18 to correctly identify animal species in images of the Island Conservation dataset, which is freely available, but not featured in any publications yet.

Characteristics of the dataset:
- ca. 120,000 high-resolution images in different sizes
- 123 camera locations from 7 islands in 6 countries
- highly imbalanced, 49 classes overall
- 60% of images are empty without an animal
- ca. 6,000 images with more than one animal

The state-of-the art for identifying the species correctly using ResNet-18 is at 98% accuracy (Tabak et al. (2018) who use a different dataset).

## How to run the code

Create a conda environment from the .yml file. Use Code/model_training.py for training + validating. Metrics are reported to local Tensorboard.

## Context of the proeject

This individual project is ongoing as part of my coursework for my Masters degree (Management & Data Science) in the winter term 2020/2021.

## References

The Island Conservation Camera Traps Dataset is hosted by the Labeled Information Library of Alexandria: Biology and Conservation (LILA BC) and is freely available [here](http://lila.science/datasets/island-conservation-camera-traps/).

The ResNet-18 implementation and pretrained weights are provided by PyTorch [here](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py).

Tabak, M. A., Norouzzadeh, M. S., Wolfson, D. W., Sweeney, S. J., Vercauteren, K. C., Snow, N. P., Halseth, J. M., Di Salvo, P. A., Lewis, J. S., White, M. D., Teton, B., Beasley, J. C., Schlichting, P. E., Boughton, R. K., Wight, B., Newkirk, E. S., Ivan, J. S., Odell, E. A., Brook, R. K., … Miller, R. S. (2018). Machine learning to classify animal species in camera trap images: Applications in ecology. Methods in Ecology and Evolution, 10(4), 585–590. https://doi.org/10.1111/2041-210X.13120
