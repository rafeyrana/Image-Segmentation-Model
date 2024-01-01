# Image Segmentation for Water Contamination Measurement

This project aims to segment water channels to measure water contamination due to trash. The model used in this project is a Convolutional Neural Network (CNN) built with Keras and TensorFlow. The model is trained to classify each pixel of an image into one of three classes: water, trash, or other.

## Overview of the Project

The project starts with importing necessary libraries and modules. It uses Keras for building and training the model, TensorFlow for mathematical operations, and other libraries like NumPy, Matplotlib, and PIL for handling and visualizing data.

The model architecture is a U-Net like structure, which is a type of CNN that is very effective for semantic segmentation tasks. The model takes an image of size 256x256x3 as input and outputs a segmented image of the same size. The output image has three channels, each representing one of the classes (water, trash, other).

The model is trained with a custom loss function called `blance_loss`, which is a weighted cross-entropy loss. The weights for the loss function are defined by the `Beta` array.

After defining and compiling the model, it is trained using the `fit` method. The model's performance is evaluated using metrics like accuracy, Intersection over Union (IoU), and Dice Coefficient.

Finally, the trained model is saved for future use.

## Detailed Explanation of the Code

The necessary libraries and modules for the project are imported.

The `Beta` array is defined, which will be used as weights in the loss function.

The `convert_to_logits` function is used to convert the model's predictions into logits, which are then used in the `blance_loss` function. The `blance_loss` function calculates the weighted cross-entropy loss between the true labels and the predicted labels.

The model architecture is defined. The model takes an image of size 256x256x3 as input and outputs a segmented image of the same size. The output image has three channels, each representing one of the classes (water, trash, other). The model is compiled with the Adam optimizer, the custom `blance_loss` function as the loss function, and accuracy as the metric.

The model is trained using the `fit` method. The training data, batch size, number of epochs, and validation data are passed as arguments to the `fit` method.

The history of the training process is printed. It includes the loss and accuracy for each epoch on the training and validation datasets.

The model's performance is evaluated on the test dataset using the `evaluate` method. The test data and batch size are passed as arguments to the `evaluate` method.

The results of the evaluation are printed. The results include the loss, accuracy, Intersection over Union (IoU), and Dice Coefficient on the test dataset.

The trained model can be saved for future use with the `save` method. The filename "model2_weights.h5" is passed as an argument to the `save` method. This line is commented out in the provided code, so it won't execute unless the comment is removed.


# Bonus Project: Image Similarity Search Web App

This project is a web application that allows users to find visually similar images in a dataset. The application is built using Streamlit, a framework for building machine learning and data science web apps.

## Overview of the Project

The application starts by importing necessary libraries and modules. It uses Streamlit for the web app, pandas for data manipulation, PIL for image processing, requests for downloading images, os and shutil for file handling, BytesIO for handling binary streams, torchvision for the pre-trained model and image transformations, Annoy for similarity search, and matplotlib for image visualization.

The application uses a pre-trained ResNet-50 model to extract features from images. The features are then indexed using Annoy, an Approximate Nearest Neighbors Oh Yeah library, which allows for fast similarity search.

The application provides a user interface where users can upload an image and specify the number of similar images they want to find. The application then performs a similarity search and displays the similar images.

## Detailed Explanation of the Project

The necessary libraries and modules for the project are imported.

The `extract_features` function is used to extract features from an image using the pre-trained ResNet-50 model.

The `show_images` function is used to display images in a grid layout using matplotlib.

The `make_dataset` function is used to download images from a CSV file, extract features from the images, and index the features using Annoy.

The `new_image_search` function is used to perform a similarity search on the uploaded image and return the paths of the similar images.

The `display_assignment_report` function is used to display the assignment report in the Streamlit app.

The `main` function is the main function of the Streamlit app. It provides a user interface where users can upload an image, specify the number of similar images they want to find, and perform a similarity search. The similar images are then displayed in the app.

The application is run by calling the `main` function when the script is run as the main module.
