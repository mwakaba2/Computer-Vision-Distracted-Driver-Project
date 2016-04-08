# State Farm Distracted Driver Detection

### Goal: Predict the likelihood of what the driver is doing in each picture

### Dataset:
<a href= "https://www.kaggle.com/c/state-farm-distracted-driver-detection/data">State Farm Distracted Driver Detection Data</a>

##### The 10 classes to predict are:
  * c0: safe driving
  * c1: texting - right
  * c2: talking on the phone - right
  * c3: texting - left
  * c4: talking on the phone - left
  * c5: operating the radio
  * c6: drinking
  * c7: reaching behind
  * c8: hair and makeup
  * c9: talking to passenger

To ensure that this is a computer vision problem, we have removed metadata such as creation dates. The train and test data are split on the drivers, such that one driver can only appear on either train or test set. 

To discourage hand labeling, we have supplemented the test dataset with some images that are resized. These processed images are ignored and don't count towards your score.
