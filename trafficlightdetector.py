import os
import glob
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import numpy as np      
import math
from time import process_time

import tensorflow as tf
from tensorflow import keras
from keras import layers  
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, mean_squared_error
from sklearn.utils import class_weight


def resize(images, new_size):
    '''  
    Resizes an image to a specified new size.
    
    @param images: 
        array of images, of shape (samples, width, height, channels)
    
    @param new_size: 
        tuple of the new size, (new_width, new_height)

    @return
        resized image
    '''
    # tensorflow has an image resize funtion that can do this in bulk
    # note the conversion back to numpy after the resize
    return tf.image.resize(images, new_size).numpy()

def resize_images(images, width, height):
    '''  
    Resizes a given array of images all to the same standard size of the height
    and width provided. The returned array of resized images is a numpy array.
    
    @param images: 
        array of images, of shape (samples, width, height, channels)
    
    @param width: 
        the new width value desried

    @param height: 
        the new height value desried

    @return
        resized array of images, (samples, new_width, new_height, channels) 
    '''
    resized_images = []
    # iterate through the passed images
    for img in images:
        # resize them and add them to the returned array
        resized_images.append(cv2.resize(img, (width, height)))
    return np.array(resized_images)

def plot_images(x, y):
    '''  
    Plots some images and their labels. Will plot the first 50 samples in a 10x5 grid
    
    @param x: 
        array of images, of shape (samples, width, height, channels)
    
    @param y: 
        labels of the images
    '''
    fig = plt.figure(figsize=[15, 18])
    for i in range(50):
        ax = fig.add_subplot(5, 10, i + 1)
        ax.imshow(x[i])
        ax.set_title(y[i])
        ax.axis('off')

def plot_class_imbalance(y):
    # Use np.unique to find unique classes and their counts
    classes, counts = np.unique(y, return_counts=True)

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts)
    plt.xlabel('Classes', fontsize = 20)
    plt.ylabel('Counts', fontsize = 20)
    plt.title('Training Class Imbalance', fontsize = 30)

def calculate_class_weights(y):
    # Calculate the class weights
    weights = class_weight.compute_sample_weight(class_weight='balanced', y=y)

    # Convert the weights to a dictionary to use with Keras
    class_weights = {i: weight for i, weight in enumerate(np.unique(weights))}

    return class_weights

def eval_model(Y_train_pred, Y_train, Y_test_pred, Y_test):
    '''  
    Evaluates the accuracy and other metrics of predicted values vs true values.
    
    @param Y_train_pred: 
        Predicted train values.
    @param Y_train:
        True train values.
    @param Y_test_pred:
        Predicted test values.
    @param Y_test
        True test values.

    @return
        numpy arrays of size (samples, width, height, channels), and size (samples) for 
        #images and thier labels.
    '''
    fig = plt.figure(figsize=[25, 8])
    ax = fig.add_subplot(1, 2, 1)
    conf_matrix = confusion_matrix(Y_train, Y_train_pred)
    ConfusionMatrixDisplay(conf_matrix).plot(ax=ax)
    ax.set_xlabel('Predicted Label', fontsize = 20)
    ax.set_ylabel('True Label', fontsize = 20)
    ax.set_title('Training Set Performance: %s' % (sum(Y_train_pred == Y_train)/len(Y_train)), fontsize = 30)
    ax = fig.add_subplot(1, 2, 2)   
    conf_matrix = confusion_matrix(Y_test, Y_test_pred)
    ConfusionMatrixDisplay(conf_matrix).plot(ax=ax)
    ax.set_xlabel('Predicted Label', fontsize = 20)
    ax.set_ylabel('True Label', fontsize = 20)
    ax.set_title('Test Set Performance: %s' % (sum(Y_test_pred == Y_test)/len(Y_test)), fontsize = 30);    
    print(classification_report(Y_test, Y_test_pred))

def pixel_comparison_classifier(images, reference_images):
    '''  
    Classifies images using a pixel by pixel comparison method.
    
    @param images: 
        array of images to be classified, of shape (samples, width, height, channels)
    
    @param reference_images: 
        dictionary with classes as keys and corresponding reference images as values

    @return
        predicted classes of the images
    '''
    predictions = []

    # Iterate through the images to be classified
    for image in images:
        min_mse = float('inf')
        predicted_class = None

        # Iterate through the reference images
        for class_name, ref_image in reference_images.items():
            # Compute the mean squared error (MSE) between the image and the reference image
            mse = mean_squared_error(image.flatten(), ref_image.flatten())

            # If the MSE is smaller than the current minimum, update the minimum and the predicted class
            if mse < min_mse:
                min_mse = mse
                predicted_class = int(class_name)

        predictions.append(predicted_class)

    return predictions

def plot_history(history):
    '''  
    load the data stored in a specifc directory
        
    @param history: 
        a trained model 

    @plot
        Two subplots which include the training history of the 
        accuracy and loss of the model vs training epochs.
    '''
    fig = plt.figure(figsize=[20, 6])
    ax = fig.add_subplot(1, 2, 1)
    plt.title("Loss vs Epochs", fontsize = 30)
    plt.xlabel("Epochs",fontsize = 20)
    plt.ylabel("Loss",fontsize = 20)
    ax.plot(history['loss'], label="Training Loss")
    ax.plot(history['val_loss'], label="Validation Loss")
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)
    plt.title("Accuracy vs Epochs", fontsize = 30)
    plt.xlabel("Epochs",fontsize = 20)
    plt.ylabel("Accuracy",fontsize = 20)
    ax.plot(history['accuracy'], label="Training Accuracy")
    ax.plot(history['val_accuracy'], label="Validation Accuracy")
    ax.legend()


def load_directory(base_path):
    '''  
    load the data stored in a specifc directory
    
    @param base_path: 
        path to the data

    @return
        numpy arrays of size (samples, width, height, channels), and size (samples) for 
        #images and thier labels
    '''
    x_path = os.path.join(base_path, 'images')
    y_path = os.path.join(base_path, 'labels')
    # find all images in the directory
    x_files = glob.glob(os.path.join(x_path, '*.png'))
    y_files = glob.glob(os.path.join(y_path, '*.txt'))

    x = []
    y = []
    
    # loop through the images, loading them and extracting the subject ID
    for fx, fy in zip(x_files, y_files):
        x.append(cv2.cvtColor(cv2.imread(fx), cv2.COLOR_BGR2RGB) / 255.0)
        with open(fy, 'r') as file:
            # Read the first line
            line = file.readline()

            # Split the line by space to get the elements
            elements = line.split(' ')

            # Get the first element and append it to the numpy array
            y.append(int(elements[0]))  # Assuming the first element is an integer     
    return np.array(x), np.array(y)

def load_data(base_path):
    '''  
    load the image data and split it into the training, test and validation sets
    
    @param base_path: 
        path to the data
    
    @param test_split: 
        the amount of images to be used in the test set from the overall data, a value > 0 and < 1

    @param val_split: 
        the amount of images to be used in the validation set from the overall data, a value > 0 and < 1

    @return
        the training, validation and test images/labels arrays
    '''
    # Loads Daisy Data
    train_X, train_Y = load_directory(os.path.join(base_path, 'processed_data/train'))
   
    # Loads Dandelion Data
    val_X, val_Y = load_directory(os.path.join(base_path, 'processed_data/val'))
    
    return train_X, train_Y, val_X, val_Y

def task_1():
    '''  
    Build and train the neural network on the Standard Transfer Leanring method. Evaluate
    its performance in the training period and its predictions in the testing period.
    
    @plot
        - An batch of the images with their correct labels
        - the loss vs epoch and accuracy vs epoch graphs of the model in training and validation
        - the confusion matrix of the problem after running the test set through the trained network
    '''

    ############################## LOADING DATA ############################################
    # Data & ML Model Parameters
    learn_rate = 0.01
    momentum = 0.0
    batch = 50
    epoch_num = 15
    channels = 3
    image_size = 128 # Play around with this (bigger is better)
    train_X, train_Y, val_X, val_Y = load_data(base_path="") # You could adjust the splits here if you wanted, maybe a bigger test set would be insigtful of the models performance

    # Bit of resizing as neural networks expect constistent sized inputs
    train_X = resize_images(train_X, image_size, image_size)
    val_X = resize_images(val_X, image_size, image_size)
    #Plot those bad boys
    plot_images(train_X, train_Y)
    plot_class_imbalance(train_Y)

    reference_images = {
    '0': cv2.resize(cv2.imread('processed_data/train/images/road32.png'), (image_size, image_size)),
    '1': cv2.resize(cv2.imread('processed_data/train/images/road397.png'), (image_size, image_size)),
    '2': cv2.resize(cv2.imread('processed_data/train/images/road155.png'), (image_size, image_size)),
    '3': cv2.resize(cv2.imread('processed_data/train/images/road85.png'), (image_size, image_size)),
    }   

    pixel_pred_y = pixel_comparison_classifier(val_X, reference_images)
    # Print confusion matrix
    conf_matrix = confusion_matrix(val_Y, pixel_pred_y)
    ConfusionMatrixDisplay(conf_matrix).plot()
    plt.xlabel('Predicted Label',size=20)
    plt.ylabel('True Label',size=20)
    plt.title('Test Set Performance: %s' % (sum(pixel_pred_y == val_Y)/len(val_Y)), fontsize = 30)
    print(classification_report(val_Y, pixel_pred_y))
    ##################################### ARCHITECTURE STUFF ###########################################
    mobile_base = keras.applications.MobileNetV2(input_shape=(image_size, image_size, channels),
                                                    include_top=False,)
    mobile_base.trainable = False

    # CLASSIFICATION HEAD
    x = layers.GlobalAveragePooling2D()(mobile_base.output)
    x = layers.Dense(1024, activation='relu', name="ClassHead")(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    classification_result = layers.Dense(4, activation='softmax')(x) # The final deciding layer

    # Initialise the architecture
    network = keras.Model(mobile_base.input, classification_result) # The first argument is the start of your model and the last is the end.
    network.summary() # Spits out some sick stuff to read if you care. (architecture)

    # Set some settings
    # tbh no idea what the optimizer does, something to do with the gradient decsent i assume
    optimizer = tf.keras.optimizers.SGD(learning_rate=learn_rate, ##### <- Play with this #####
                                        momentum=momentum,       ##### <- Play with this #####
                                        nesterov=False)
    network.compile(optimizer=optimizer, loss=['sparse_categorical_crossentropy'], metrics=['accuracy']) # That loss argument is pre important, categorical_crossentropy is for multi classification tasks
    class_weights = calculate_class_weights(train_Y)
    # We timing this fool
    time_1 = process_time()
    # Train the network (just the classification head)
    history = network.fit(train_X, train_Y, epochs=epoch_num, batch_size = batch, validation_data = (val_X, val_Y), class_weight = class_weights) # Try increase the batch size and increase the epochs and see the performance
    time_2 = process_time() # fool timed
    train_time = str(time_2 - time_1) 
    print("Time to train model 1: " + train_time)

    #Saves the trained weights
    #network.save_weights('Trained_Weights/Model1')

    ########################################## Predict and Evaluate ###########################################################
    # Run Inference
    train_pred_Y = network.predict(train_X)
    test_pred_Y = network.predict(val_X)

    # Plot Loss & Accuracy History
    plot_history(history.history)

    # This stuff is converting probabilities into a yes its this class.
    # Convert probabilities to class labels
    train_pred_Y = np.argmax(train_pred_Y, axis=1)
    test_pred_Y= np.argmax(test_pred_Y, axis=1)

    # Also make sure your true labels are class labels and not one-hot encoded
    train_Y = np.argmax(train_Y, axis=1) if train_Y.ndim > 1 else train_Y
    test_Y = np.argmax(val_Y, axis=1) if val_Y.ndim > 1 else val_Y

    # Now you can evaluate the model
    eval_model(train_pred_Y, train_Y, test_pred_Y, test_Y)

    # This makes the plots visable.


    plt.show()

def task_2():
    '''  
    Build and train the neural network on the Accelerated Transfer Leanring method. Evaluate
    its performance in the training period and its predictions in the testing period.
    
    @plot
        - the loss vs epoch and accuracy vs epoch graphs of the model in training and validation
        - the confusion matrix of the problem after running the test set through the trained network
    '''
        ############################## LOADING DATA ############################################
    # Data & ML Model Parameters
    learn_rate = 0.01
    momentum = 0.0
    batch = 50
    epoch_num = 10
    image_size = 128
    channels = 3
    train_X, train_Y, val_X, val_Y = load_data(base_path="")

    # Bit of resizing as neural networks expect constistent sized inputs
    train_X = resize_images(train_X, image_size, image_size)
    val_X = resize_images(val_X, image_size, image_size)

    #Plot those bad boys
    plot_images(train_X, train_Y)


    ##################################### ARCHITECTURE STUFF ###########################################
    mobile_base = keras.applications.MobileNetV2(input_shape=(image_size, image_size, channels),
                                                    include_top=False,)
    mobile_base.trainable = False

    # CLASSIFICATION HEAD
    input_layer = layers.GlobalAveragePooling2D()(mobile_base.output)
    x = layers.Dense(1024, activation='relu', name="ClassHead")(input_layer)#(x) #kernel_regularizer=l2(0.01)
    x = layers.Dropout(0.5)(x)
    classification_result = layers.Dense(4, activation='softmax')(x) # The final deciding layer

    # Initialise the architecture
    network = keras.Model(mobile_base.input, classification_result) # The first argument is the start of your model and the last is the end.
    network.summary() # Spits out some sick stuff to read if you care. (architecture)



    # Define the base model and a GlobalAveragePooling2D layer
    base_network = keras.Model(mobile_base.input, mobile_base.output)
    pooling_layer = layers.GlobalAveragePooling2D() # Not too sure about this line but it fixed the problem.

    # Create the embeddings, applying the pooling operation
    train_embeddings = pooling_layer(base_network.predict(train_X))
    val_embeddings = pooling_layer(base_network.predict(val_X))

    classification_head = keras.Model(input_layer, classification_result)


    # Set some settings
    # tbh no idea what the optimizer does, something to do with the gradient decsent i assume
    optimizer = tf.keras.optimizers.SGD(learning_rate=learn_rate,
                                        momentum=momentum,
                                        nesterov=False)
    classification_head.compile(optimizer=optimizer, loss=['sparse_categorical_crossentropy'], metrics=['accuracy']) # That loss argument is pre important, categorical_crossentropy is for multi classification tasks

    # We timing this fool
    time_1 = process_time()
    # Train the network (just the classification head)
    history = classification_head.fit(train_embeddings, train_Y, epochs=epoch_num, batch_size = batch, validation_data = (val_embeddings, val_Y))
    time_2 = process_time() # fool timed
    train_time = str(time_2 - time_1) 
    print("Time to train model 1: " + train_time)

    #Saves the trained weights
    classification_head.save_weights('Trained_Weights/Classification_head')

    ########################################## Predict and Evaluate ###########################################################
    # Run Inference
    train_pred_Y = classification_head.predict(train_embeddings)
    test_pred_Y = classification_head.predict(val_embeddings)


    # Plot Loss & Accuracy History
    plot_history(history.history)


    # This stuff is converting probabilities into a yes its this class.
    # Convert probabilities to class labels
    train_pred_Y = np.argmax(train_pred_Y, axis=1)
    test_pred_Y= np.argmax(test_pred_Y, axis=1)

    # Also make sure your true labels are class labels and not one-hot encoded
    train_Y = np.argmax(train_Y, axis=1) if train_Y.ndim > 1 else train_Y
    test_Y = np.argmax(val_Y, axis=1) if val_Y.ndim > 1 else val_Y

    # Now you can evaluate the model
    eval_model(train_pred_Y, train_Y, test_pred_Y, test_Y)

    # This makes the plots visable.
    plt.show()

if __name__ == "__main__":
    task_1()
    #task_2()