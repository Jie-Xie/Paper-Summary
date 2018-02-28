# Learning Video Object Segmentation with Visual Memory

## Task
Create binary segmentation of objects that moved in at least one frame with video frames and estimated optical flows without any ground truth annotations.

## Model Architecture

### Appearance Network
Input: RGB frame (3 x w x h) ->
Pretrained model (freezed): up to the fc6 layer of largeFOV version of the DeepLab network pretrained on PASCAL VOC 2012 ->
(1 x 1 convolutional layer + tanh) x 2
Output: (128 x w/8 x h/8)

### Motion Network
Input: optical flow ->
Pretrained model (freezed): MP-Net (w/4 x h/4)->
2 x 2 max pooling layer with stride 2
Output: motion prediction (w/8 x h/8)

### Visual Memory Module
Input: concatenation of appearance and motion output ->
Bidirectional ConvGRU: formulation shown in page 4 (all 6 convolutional layers are 7 x 7); concatenate activations from the two directions (64 x w/8 x h/8) to produce a (128 x w/8 x h/8) output for each frame ->
3 x 3 convolutional layer (64 x w/8 x h/8) ->
1 x 1 convolutional layer + softmax
Output: pixel-wise segmentation 

## Dataset
* Training: DAVIS
* Test: DAVIS, FBMS, SegTrack-v2

## Data Preparation
Batch: size 14; randomly select a video and then select 14 consecutive frames
Augmentation: 
1. random cropping and flipping
2. duplicate the first/last five frames to simulate stop-and-go scenario to create additional training sequences
Proportion of batches with additional sequences: 20%

## Training
Weight initialization: xavier method for non-ConvGRU convolutional layers
Gradients clipping: [-50, 50]
Loss: binary cross-entropy loss
Optimization: backpropagation through time with RMSprop, lr = 1e-4, weight decay = 0.005
Iterations: 30000

## Post-processing
fully-connected CRF
