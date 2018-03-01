# [Real-Time Recurrent Regression Networks for Visual Tracking of Generic Objects](https://arxiv.org/pdf/1705.06368.pdf)

## Task
Track a generic object throughout a video with only the bounding box of the initial frame.

## Implementation 
[re3-tensorflow](https://gitlab.cs.washington.edu/xkcd/re3-tensorflow)

## Model Architecture

### Appearance Network
* Input: a pair of crops among which the first is centered at the object's location in the previous frame and the second at the same location in the current frame; both are padded to be twice the size of the object's bounding box and warped to 227 x 227.
* Convolutional pipeline: CaffeNet with skip connections (1 x 1 conv layer + PReLU + flatten layer) after norm1, norm2 and conv5 layer with 16, 32, and 64 channels respectively. 
* Late fusion: concatenate features including CaffeNet output and outputs of the three skip connections.
* Fully-connected layer: fc6 with 2048 units

### Recurrent Network
* Input: output of the appearance network
* LSTM: two layers of 1024-unit factored LSTM with peephole connections (formulation shown in page 3)
* Fully-connected layer: 4 outputs
* Output: top left and bottom right corners of the new bounding box

## Training
* Weight initialization: MSRA initialization method for new layers
* Loss: L1 loss
* Optimization: ADAM, initial learning rate = 1e-5 which decreases to 1e-6 after 10000 iterations for another 200000 iterations
* Maximum sequence length: 32
* Scheme: initially train with two unrolls and mini-batch size = 64 (meanwhile use ground-truth crops); after the loss plateaus, double unrolls and halve mini-batch size (meanwhile increase probility of using predicted crops) until unrolls of 32 and mini-batch size = 4 




