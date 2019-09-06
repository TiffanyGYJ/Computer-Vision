# S³FD: Single Shot Scale-invariant Face Detector
A PyTorch Implementation of Single Shot Scale-invariant Face Detector converted to Face Recognition system.

Code based on : https://github.com/clcarwin/SFD_pytorch

## Train 
```
python3 Finetuning_clean.py
```

Notes and explanation at https://deshanadesai.github.io/notes/PyTorch

For thd pre-trained model weights, please unzip the 7z file : s3fd_convert.7z .

## Gender Recognition Instructions:

The end result is to classify a picture to either male or female. You need to familiarize yourself with the architecture of S3fd and make sure the original s3fd.py (or is it net_s3fd.py) makes sense to you.

Having done that, you need to add Gender Classification Layers to output of detection layers. As an example, I have added:

```
        self.conv4_3_norm_gender = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
```
on line 59 of the s3fd.py file. In the forward function too, you need to make sure the change is reflected:

 ```
         gen2 = self.conv4_3_norm_gender(f4_3)
```
on line 135 of the s3fd.py file. Finally return the face classification score outputs and gender classification outputs (you do not need the bounding box location outputs but can return and not use them later).

```
        return [cls4,gen1]
```

You need to add such gender classification layers for each of the outputs of the detection layers. You can skip the first and last because the scale of our picture does not match the face proposals detected at those scales.

-----

Now, Go to the Finetuning_clean.ipynb and make the following changes:

1) After turning the Gradients of all the layers to False (i.e. "don't compute"), fetch the weights and bias of the corresponding Face Classification layer and make your gender classification layer equal to that.
```
for param in myModel.parameters():
    param.requires_grad = False

# Accessing the weights of a classification layer:
myModel.conv4_3_norm_mbox_conf.weight[0]

# WE ONLY WANT THE FACE WEIGHTS (NOT THE NON FACE WEIGHTS). So pick up the weights from dimension 0.
# Store them in a tensor, add a dimension to the new tensor.
# Concatenate the tensor with itself so that you have a 2 channel tensor (for gender = F and gender = M).
# Add this new tensor to the Gender layer's weight. Make sure it has dimension N x C x H x W = N x 2 x H x W. (N is the number of images in the batch)
# Do the same for all Gender classification layers.
# Pytorch documentation and forums are very helpful to figure out how these things can be implemented.
# Print the weights and make sure they are the same as the corresponding Face Classification layer's first channel.
# You can play around with random initialization of weights too or different initializations. Whatever you like.
```

2) Now while training, apply softmax to the gender classification layers the same way it is applied in the face classification layers :

```
    for i in range(len(olist)/2): olist[i*2] = F.softmax(olist[i*2], dim = 1)
```

in https://github.com/deshanadesai/SFD_pytorch/blob/master/test.py

This code was explained in class.

3) Based on your returned output from the forward() function, fetch the gender layer outputs. (The same was as ocls and oreg are fetched). Iterate through all the cells with (hindex, windex) for the face score outputs (the same way as is done in line 33, 34 in test.py). If the score of this region proposal being a non-face is less than 0.05, ignore it. (same way as is done in test.py). Extract the gender scores if the score of this region proposal being a non-face was greater than 0.05, (same way as you extract the loc in line 37 of test.py). Ignore the rest of location based processing in test.py with variances, decoders, priors etc. Append this gender prediction scores (1x2 dimension) to a list.

4) For each of the predictions in gender prediction list: calculate the loss and add it to the total loss for this iteration.

Note that the loss is not computed over ALL the gender predictions, it is only computed over the gender predictions corresponding to the region proposals that were identified as a FACE. Imp to understand this. Rest is pretty much the same. Use BCELoss() (Binary cross entropy loss) but play around with different loss functions.

Hope this helps!

