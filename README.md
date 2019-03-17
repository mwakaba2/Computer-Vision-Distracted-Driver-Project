# State Farm Distracted Driver Detection

### Goal: Predict the likelihood of what the driver is doing in each picture

### [Final Report](https://github.com/mwakaba2/Computer-Vision-Capstone-Project/blob/master/CapstoneProject.pdf)

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

Removed metadata such as creation dates. The train and test data are split on the drivers, such that one driver can only appear on either train or test set. 

### Documentation
Capstone_Project.pdf : final report of the capstone project
#### classifier/
  * classifier.py : run ``python classifier.py`` to create classifiers and save score and submission

#### data_analysis/
  * analyze_data.py : run ``python analyze_data.py`` to obtain stats about the training data
  * hog_visualizer.py : run ``python hog_visualizer.py`` to create and save hog descriptors for a random subject
  * surf_matching.py : run ``python surf_matching.py`` to obtain surf descriptors and match key points for two images for each class
  * plot_results.py : run ``python plot_results.py`` create html that shows a visualization of the classifiers performance according to feature set
  * output/ : 
   * hog_images.png : hog descriptor example for report
   * surf_group1.png : surf descriptor example for report 
   * surf_group2.png : surf descriptor example for report 
  
#### imgs/
  * test/ : test images that needs prediction
  * train/ : 10 folders of labeled images
  * driver_imgs_list.csv : csv that contains image name, label, subject id

#### objects/
  * test/ : pickled objects for test images
  * train/ : pickled objects for train images

#### submissions/
  * base_submission.csv : predictions for using haralicks features only and Logistic Regression
  * combined_all_submission.csv : predictions for using haralicks, surf and lbps features and Logistic Regression
  * combined_submission.csv : predictions for using haralicks features and lbps features and Logistic Regression
  * lbps_submission.csv : predictions for using lbps features only and Logistic Regression
  * surf_submission.csv : predictions for using surf features only and Logistic Regression
  * final_model_results.html : visualization of Logistic Regression performance based on feature set
  * results_LR.image.txt : Score and best parameters for each feature set combination for model Logistic Regression
  * results_RF.image.txt : Score and best parameters for each feature set combination for model one vs rest with Random Forests
 
