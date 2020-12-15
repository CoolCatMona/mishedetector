![Mishe1](https://imgur.com/VV1FaoI.png) ![Mishe2](https://imgur.com/oxFJfKM.jpg)

# Siamese Neural Networks, One Shot Learning, and Facial Recognition

Originally created as a small project to familiarize myself with concepts of One-Shot Learning, Siamese Neural Networks, and using those concepts to implement a small facial recognition application.
This is a facial recognition application that predicts wether an image of a face _is_ or _is not_ a **Mishe**, a Mishe, described as "A girl with cheerful ears and a tail", is a 3D character created by [Ponderonium Institute](https://ponderogen.booth.pm/items/1256087).
A higher number [0, 1] implies that the image given _is_ of a Mishe.

## Siamese Neural Networks

A **Siamese Neural Network** (SNN) is a type of neural network architecture that contains two or more identical subnetworks; meaning they have the _same_ configuration and hyperparameters.
These are used to find the similarity of the inputs by comparing the resultant feature vectors from each architecture. We can find the euclidian distance betweent he two feature vectors to differentiate them.

[ put a cool picture of the architecture here ]

This SNN is composed of a **Convolutional Neural Network** (CNN), which you can read more about [here](https://en.wikipedia.org/wiki/Convolutional_neural_network). Typically, a network like this would be used for image classification of one or more labels, however, since we are building a binary classification, we can halve the size of the network.
In a typical network, each class image would need its own embedding. Such as the classic [cats vs dogs](https://towardsdatascience.com/image-classifier-cats-vs-dogs-with-convolutional-neural-networks-cnns-and-google-colabs-4e9af21ae7a8) problem.
In our case, the size is halved as we are simply predicting the probability that an image **is** of class **X**

## One Shot Learning

Typically used in computer vision problems, One-Shot Learning is used to learn information about objects from only a few training samples. Usually we would expect much larger datasets for computer vision problems.
In our example, we want to be able to recognize that two faces are of the same type of person. So, given that the network is supplied with photos of two of the same type of person, we would expect the model to predict with a high probability that the images _are_ of the same person. Inversely, if the model is supplied with two photos of a _different_ type of person, we would equally expect the model to predict that the images _are not_ of the same person.

## Design Choices and Implementation

- snn.py Contains implementation of the SNN itself as well as a custom method to train the model
- process_images.py contains implemntation for loading images from folders, converting the images into a format that our model can use, and splitting the images into training and test sets
- main.py is the main file and should be ran to see results.

### Data Collection

Data for Anchors and Positives were collected myself by taking various pictures of users in the game of [VRChat](https://hello.vrchat.com/) who were using a Mishe as a base model for their avatar. The pictures were cropped to 256x256 pixels on 3 separate RGB channels.
Because the game allows for custom model creation, it is possible that the model suffers as a result of Mishe having different colored hair or eyes. Although I expect this will actually increase the accuracy of the model.

### Regarding the SNN

The implementation of the SNN was fairly straightfowrard up to the custom training step and hyperparameter choices. Various other places were referenced to create this and may use concepts that I do not go into in-depth such as Layer-wise Learning rates, momentum, and l2-regularization penalties.

### Triplet Loss

In order to train this network, we utilize the **Triplet Loss** function. That is the data is divided into **Anchor**, **Positive**, and **Negative** pairs
- $A$nchor is an image of our base model's face
- $P$ositive is _another_ image of our base model's face
- $N$egative is an image _not_ of our base model

Each training example becomes a triplet of (A, P, N)
Assuming we have a neural network model that can take a picture as input and output an embedding of the picture, the triplet los for the xample is

<img src="https://render.githubusercontent.com/render/math?math=max(||f(A_i) - f(P_i)||^2 - ||f(A_i) - f(N_i)||^2 + \alpha, 0)">


The cost function is defined as the average of the triplet loss:

<img src="https://render.githubusercontent.com/render/math?math=\frac{1}{N}\sum_{i=1}^{N} max(||f(A_i) - f(P_i)||^2 - ||f(A_i) - f(N_i)||^2 + \alpha, 0)">

Where <img src="https://render.githubusercontent.com/render/math?math=\alpha"> is a positive hyperparameter. So we can see that <img src="https://render.githubusercontent.com/render/math?math=||f(A_i) - f(P_i)||^2"> is low for _similar_ embeddings, and high for _dissimilar_ embeddings.

## TODO
- Acquire more data
- Hyperparameter tuning
- Image Augmentation (for training)
- Differentiate between multiple classes

## Credits

I owe the following sources credit for assistnace in creating this application:

- [Siamese Neural Networks for One-Shot Image Recognition Koch, Zemel, Salakhutdinov](http://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
- [Siamese-Networks-for-One-Shot-Learning](https://github.com/tensorfreitas/Siamese-Networks-for-One-Shot-Learning)
- [The Hundred-Page Machine Learning Book by Andriy Burkov](http://themlbook.com/)
