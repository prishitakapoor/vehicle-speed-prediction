# vehicle-speed-prediction

<p align="center">
 <img src=".\imgs\splash.png" width="480">
</p>

This is my solution to comma.ai's programming challenge which you can download [here](http://geohot.com/speed_challenge.tar.gz). The goal is to predict the speed of a car at each frame of a video.

## Data

- `drive.mp4`: the video shot at 25 fps. Equivalent to 8616 frames of size 640x480.
- `drive.json`: json file containing 8616 pairs of (`exact_picture_time`, `speed_in_meters_per_second_at_that_time`). Exact speed values were ground truthed from the car speedometer.

## Requirements

- ffmpeg
- python 3+
- keras
- scikit-learn

## File Structure

- `preprocessing.py`: run this script to convert the video to a set of frames and then the frames to a set of numpy arrays. Also reads in the ground truth speeds.
- `main.py`: extract the features using a pretrained vgg-16. You only need to do this once because the features are dumped as pickle files. Additionally, train a ridge regression model on the extracted features, test on the test data and return the MSE error.

## My Approach

I decided to take this challenge right after completing Stanford's [CS231n](http://cs231n.stanford.edu/), so I knew right away I wanted to leverage a ConvNet in my solution. I had 2 problems though:

- **first**, the dataset (~8.5k images) was not large enough to train a decent enough network. I could have used some aggressive data augmentation but this would prove to be computationally expensive.

- the **second** and main bottleneck is that I don't own a decent GPU and my RAM is not large enough (*I was not familiar with cloud computing services like AWS at the time. Shame on me!*) so I knew it would be smart to use some sort of transfer learning.

My ultimate decision was to use a VGG16 pretrained model and extract the features of the images at the earlier layers of the network.

<span style="color:red">**Why earlier?**</span> Well, my dataset is different than the one vgg16 was trained on: in fact vgg was trained on imagenet which is a dataset of animals, cars and planes. Since a convnet detects low level generic features at the early layers of the network, (i.e. edge detectors or color blob detectors) and as you move up it combines those to detect more high-level/training-data specific structures, I decided I would use `block2_pool` of vgg16.

<p align="center">
 <img src=".\imgs\vgg16.png" width="330">
</p>

As you can see in the image above, `block2_pool` is located in the early layers of the network.


## Preprocessing Step

- Converted the drive.mp4 to 8616 640x480 images using the following bash command:

```
ffmpeg -i drive.mp4 -r 25 './frames/img%04d.jpg'
```

- Working with 640x480 images is a bit challenging since I only have 8 gb of RAM so I downsampled them to a set of 8616x50x50 and 8616x200x200. Then I split the images into 90% train and 10% data and saved them as HDF5. This meant I had 7754 `X_train` data points and 862 `X_test` data points. Below is an example 50x50 image.

<p align="center">
 <img src=".\imgs\50x50.png" width="200">
</p>

- Read in the JSON file, got rid of the first column and just saved the speeds. This meant that `y[0]` corresponds to the speed of the car in the zeroth image.

- Note that I shuffled both X and y before splitting into the train and test set so that the testing set for example would not consist solely of images at the end of the video. This would help prevent overfitting.

For the full code of the preprocessing step, open `preprocessing.py`.

## Model Training

I took `X_train` and `X_test`, passed them through VGG16 right up until `block2_pool` and got the output `train_features` and `test_features`. Those had a dimensionality of 18432.

I then trained a linear classifier (specifically linear regression with regularization) on those features and got an MSE of 1.76. Could have used an ensemble of models, or the more powerful xgboost but I thought linear regression did pretty good and I got under the MSE of 4 so it was not necessary to go further. Plus by Occam's razor, it's usually more favorable to go with the less complex solution, not to mention that this sort of solution had to work on embedded systems so speed would be primordial.

I experimented with the 200x200 pixels, it took way longer on my computer since I had to resort to swap memory, I only used 2600 datapoints and was able to get the error down to 1.19. If I use the full train data (7754), I'm sure the error can be reduced to below 1.

To view the code for this part, open `main.py`.

Here's an image of the MSE calculated for 200x200 images on the held out test data:

<p align="center">
 <img src=".\imgs\mse.png" width="380">
</p>

## Misc.

Just for the heck of it, here's a t-sne visualization of the training features extracted by the convnet. Looks like a bunch of colored spaghetti if you ask me!

<p align="center">
 <img src=".\imgs\tsne.png" width="420">
</p>
