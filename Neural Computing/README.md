# Intel & MobileODT Cervical Cancer Screening

## Introduction

Cervical cancer is a disease that can be prevented if discovered in the pre-cancerous stage. Due to a lack of knowledge in this area, the most important issue is to find the appropriate method to treat each person’s cancer. Indeed, depending on the physiology of the patient, the needed treatment may not be the same. A mistake in the choice of treatment can be fatal to the patient’s recovery. Unfortunately, these mistakes often occur especially in rural areas where there are few experts in the field.

MobileODT has developed a system to help healthcare providers in this field make the best possible treatment decisions based on the physiology of the individual in rural settings. However, analysis can take time, and the company would like to be able to give a response in real-time. This would allow the type of treatment to be chosen according to the type of cervix a woman has. The lack of experts in this field and the need to have the answer in real-time are the two main reasons for the use of machine learning.

## Literature survey

Prior to the Kaggle competition, researchers were more focused on creating algorithms that could predict the presence or absence of cancer. In the article ‘Comparison of machine and deep learning for the classification of cervical cancer based on cervicography images’, they had tried to develop a method to predict cervical cancer status thanks to machine learning and deep learning. After selecting the best features by using the LASSO, they trained three machine learning algorithms: XGBoost, SVM, and Random Forest. They applied a transfer learning algorithm for Deep learning, using the Res-Net 50 algorithm pre-trained on ImageNet. The results are quite good, especially for the deep learning model which completely outperforms the machine learning models, as we would have expected, with an accuracy of 0.91.

On the other hand, most of the cervical type predictions, are from papers written following the Kaggle challenge. Chaitanya Asawa et al. used Convolution Neural Network, but also transfer learning algorithms (ResNet and Inception V3) to predict cervix types. The results show that even if the training set’s accuracy can be high, the validation set’s one is not, which is proof of overfitting. Finally, the Inception-V3 has the best results on the validation set, with an accuracy of 60.47%. Lei Lei et al. experimented with numerous transfer learning algorithms with the tuning of hyperparameters and came to the same conclusion that Inception-V3 has better results on the validation set, with an accuracy of 70%. Some have tried to improve their algorithms through better data preprocessing. Indeed, a good prediction requires the acquisition of a good dataset, for that, a segmentation of the images can be a good idea, to concentrate on the good parts of our images in the inputs of the algorithm. Greenspan et al. propose a method to segment the interesting parts of cervical images.

In this work, I will create a two-layer CNN in contrast to the existing four-layer ones to reduce the training time. In addition, I will add a dropout to avoid overfitting. On the other hand, for transfer learning, I will also use Inception V3, however, I will not train the same layers.

## Data processing approach

### Data pre-processing

From a legal point of view, this data can only be used within the competition. The images may not be transmitted, duplicated, or published so that people outside the competition have access to them. In addition, each participant agrees to notify Kaggle if any breach of these rules is made by an individual. A real issue would be to be able to find the patients in the images available. Even if any information about the individuals is provided, it is our ethical duty not to look for more information and to focus on our task of predicting the type of cervix from these images.

For this challenge, we have images of three possible cervical types, 250 images for type 1, 781 for type 2, and 450 images for type 3. To submit a prediction in the Kaggle challenge, we need to successfully predict the type of 4018 images, giving for each, the probability that it is a type 1, 2, or 3. The dataset is unbalanced, so it will be important to pay attention to the results obtained for each type when making predictions.

The first step is to download the data. Once the data is available, the
pre-processing can begin. The first operation is to translate the image into
an array. To do this, the size of the images is arbitrarily chosen to be 80
by 60, with the idea of not having an array that is too large (which would lead to a too-long training without GPU) and not too small either, so as not to lose important details. As the images were not initially square, rectangle size was taken for all of them in order not to distort the images too much. When we look at the images below, for the different types, we notice that the diversity of lights, angles of shots, backgrounds is notable, which can make the prediction of types complicated.

To facilitate prediction, it can be interesting to obtain simpler images by removing the noise that may be present. To go further, the idea, as presented in the literature survey, is to segment the images to keep only the important information to predict the type of cervix. We can see from the segmented images that this works relatively well for type 1. However, for many of type 2 and 3 pictures, because the cervix is less open, our segmentation does not seem to work as well.

### Train, validation, and test set

The second step is to create our train and validation set to train our models. The X array is obtained thanks to the pre-processing explained before. Thanks to the names of the folders, it is easy to create the array with the number of the type of cervix corresponding to each image. This allows us to create y as a list of 1's, 2's and 3's, which is going to be converted into a matrix of three columns of 0's, with one 1 per row in the column corresponding to the type’s number. To obtain the train and the validation set, these two arrays are separated with a proportion of the validation set of 0.33.

The test set is created by pre-processing the available data by converting the images into size arrays (80, 60, 3).

## Machine Learning Approaches

Several models chosen thanks to the literature previously read will compete to perform the prediction. First, a CNN is created and proceeds with a tuning of the hyperparameters. Then, data augmentation will be used to train our model on more data, in the hope that this will reduce overfitting. Finally, according to the literature survey, transfer learning will be performed using the pre-trained Inception V3 algorithm, which seems to give the best results.

### Convolutional Neural Network

The first model implemented is a convolutional neural network composed of two convolutional layers. CNN models are widely used in computer vision as convolutional layers allow the extraction of important features from images without further assistance.

There are two stages for our convolutional neural network. The first one is feature extraction, during which we take the images to extract some interesting features. Two identical sets of layers for the first step are used. The convolution layer is employed to extract features and then, we apply max pooling to reduce the dimensionality, and a dropout layer to reduce overfitting. Feature extraction works by doing linear operations into the input using multiplication of weights with the input. On the other hand, the max-pooling (2x2) layer reduces the dimension of the input by taking the maximum value for each square 2x2 of the input. To finish, the dropout layer (0.5) decreases overfitting by randomly setting to zero the output of half of the neurons during training, so the others need to adapt.

Once some features are extracted, we move to the second stage of our network to classify the image. The layer ‘Flatten’ changes the input into only one dimension so that we can use the dense layer to obtain 256 results and then 64. In the end, a dense layer is used as the output layer with three units because we want to classify into three classes, and the activation ‘softmax’ is used given that we want to do classification with more than two possibilities.

Once the initial model has been trained, the hyperparameters will be tuned to find the best performing model. To do this, since trying all possible combinations would be too resource-intensive,
a ‘random grid search’ is applied, which allows us to randomly select the competing combinations. This way we find the best possible combination to create our new model which we train on our data.

### Data Augmentation

When training our models, overfitting is exhibited, so to try and avoid this it would be interesting to train our models on more data. To do this, data augmentation is performed, using the keras ‘image data generator’. It allows the generation of new image data from original ones in real-time. This allows us to train our tuned model on more data.

### Transfer Learning

Unfortunately, this does not solve our overfitting issue. Our idea is therefore to use a model previously trained on many data to transfer this knowledge to our own problem. For this purpose, the Inception V3 model is chosen because it seems to have given the best results, previously trained on the Image Net data set. Unlike other models, inception does not use deep convolution layers which most of the time lead to overfitting, but rather parallel layers. To use this template, our images need to be resized first by (299, 299, 3). In the model, the layers after the first concatenation (‘mixed0’) are removed because a too deep network on our data leads directly to overfitting. Then, we add our classification stage, previously used on our CNN. Only this part of the model is going to be trained.

## Results and future work

Unfortunately for all our models if they are trained on too many epochs, they end up overfitting, as the following graphs show for the CNN with data augmentation. The loss of the train set continues to decrease while the loss of the validation set decreases and then increases again, evidence of overfitting.

Furthermore, by comparing the results of each of our models creating a confusion matrix for each on the validation set, we notice that our models predict
predominantly type 2, which is due to the fact that we have more type 2 images than others. This shows us again that our models do not really learn generalizable things from the data, but rather learn by heart the results of the train set.

There are several reasons for our results which are unfortunately not exceptional in predicting cervical type. Firstly, due to a lack of computer power and available memory, we could not use all the available images to train our models. Therefore, the lack of data may be at the origin of the overfitting
of our models, which only learn the images by heart without finding distinct features that allow them to be classified with certainty. But also, the fact that our dataset is very unbalanced influences the models to predict mostly type 2. Secondly, as we noticed when analysing the data, the images are not all of the same quality and do not have the same viewing angles or colours. This can lead to confusion from the models, who will focus on details rather than what is interesting in the images.

Thus, to improve the results, we could first simply use all the images, which would allow us to increase our data set, and thus favour the generalisation of our algorithms. In the same idea, having a powerful computer would allow testing all the possible combinations to do the fine-tuning of the hyperparameters of our CNN. This would make it possible not to miss a combination that could be better than the one chosen by the random search. On the pre-processing side, it can be improved by developing better image segmentation. Indeed, in many images, a background that does not correspond to the cervix is present. Thus, succeeding in segmenting the images to keep only the part representing what interests us would allow a better focus of our algorithms on the important features.
