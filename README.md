# wildfire-localization-cnn

Project code for CS230.  
  
*isolate_superpixels_cs230.py splits data and runs superpixel segmentation  
*baseline_SPinceptionV1OnFire_cs230.py uses model from Dunnings and Breckon (2018, repo: fire-detection-cnn) to predict on my test set as a baseline    
*inceptionV1OnWildfire_cs230.py implements model fine tuning  
*predict_SPinceptionV1OnFire_cs230.py predicts using the fine tuned model and computes evaluation metrics F1 and IoU score  
*saliency_map_cs230.py generates saliency maps for a given image  
  
The dataset used in this project is the Corsican Fire Database (http://cfdb.univ-corse.fr).