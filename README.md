# Aerial-Semantic-Segmentation
I aim to develop a semantic segmentation model for aerial imagery that can accurately classify each pixel into meaningful land cover categories, using the Fastai library and a U-Net architecture with a ResNet-34 backbone.

## Motivation and Problem Significance
Semantic segmentation of aerial imagery allows for precise land cover classification, enabling better decision-making across many industries. From tracking urban growth and mapping flood damage to monitoring deforestation, this technology has real-world impact. By training a segmentation model capable of distinguishing between multiple land cover classes, this project contributes toward faster, more accurate, and cost-effective analysis of large-scale aerial datasets.

## Key Technologies
* Fastai: Deep learning framework that streamlines data preprocessing, model creation, and training workflows.
* PyTorch: Core deep learning library providing computational backbone for model training.
* U-Net Architecture: Encoder-decoder convolutional neural network designed for pixel-wise image segmentation.
* Resnet34: Pretrained convolutional neural network used as the encoder in U-Net for feature extraction.

## Approach
* [Dataset](https://www.tugraz.at/index.php?id=22387): contains 400 publicly available aerial images and their corresponding segmentation masks.
* Mapped RGB mask colors to class indices using a color-to-class dictionary and applied a DataBlock pipeline with ImageBlock and MaskBlock to handle paired image-mask loading.
* Implemented a U-Net model with a ResNet-34 backbone pre-trained on ImageNet.
* Chose CrossEntropyLossFlat for handling multi-class segmentation and monitored performance with the Dice coefficient metric to capture segmentation quality.
* Experimented with batch sizes and learning rates to optimize GPU usage and prevent out-of-memory errors.
* Visualized predictions alongside original images and masks to qualitatively assess segmentation accuracy.

## Future Work
The next step would be to experiment with progressive resizing. I fine tuned models separately as I was curious about the effect of image resolution on the learning rate of the model, but what would be really interesting would be to progressively resize images from 128x128 to 256x256 to 512x512, and compare model performance at each step.
