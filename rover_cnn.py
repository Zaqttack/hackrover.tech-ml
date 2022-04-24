import numpy as np
import pandas as pd
import math
import string
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau

from sklearn.preprocessing import LabelBinarizer


EPOCHS = 4
CASES = {index:letter.upper() for index,letter in enumerate(string.ascii_lowercase)}
MODEL_PATH = "piccolo/model.pickle"
HISTORY_PATH = "piccolo/history.pickle"


# creates test/train split
def train_test_split():
    train_df = pd.read_csv("./data/sign_mnist_train.csv")
    test_df = pd.read_csv("./data/sign_mnist_test.csv")
    y = test_df['label']
    y_train = train_df['label']
    y_test = test_df['label']
    del train_df['label']
    del test_df['label']
    x_train = train_df.values
    x_test = test_df.values

    return x_train, y_train, x_test, y_test, y


def preprocessing(x_train, y_train, x_test, y_test):
    # basically becomes a correct or not problem
    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.fit_transform(y_test)

    # normalizes range of RGB vals
    x_train = x_train / 255
    x_test = x_test / 255
    x_train = x_train.reshape(-1,28,28,1)
    x_test = x_test.reshape(-1,28,28,1)

    return x_train, y_train, x_test, y_test


def augment_data(x_train):
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(x_train)

    return datagen


def train_cnn(x_train, y_train, x_test, y_test, extra_data, load_from=None):
    if not load_from:
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

        model = Sequential()
        model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
        model.add(BatchNormalization())
        model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(Flatten())
        model.add(Dense(units = 512 , activation = 'relu'))
        model.add(Dropout(0.3))
        model.add(Dense(units = 24 , activation = 'softmax'))
        model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
        model.summary()

        history = model.fit(extra_data.flow(x_train, y_train, batch_size = 128),
                            epochs = EPOCHS, validation_data = (x_test, y_test) ,
                            callbacks = [learning_rate_reduction])
    if load_from:
        # checks if the read fails
        success = read_model_history(model_path=load_from[0],
                                            history_path=load_from[1])
        if success:
            model, history = success[0], success[1]
        else:
            return train_cnn(x_train, y_train, x_test, y_test, extra_data)

    accuracy = model.evaluate(x_test, y_test)[1] * 100

    predictions = model.predict(x_test)

    return model, predictions, history, accuracy


def plot_preprocessed_images(images):
    width = 5
    height = math.ceil(len(images) / width)

    f, ax = plt.subplots(2,5) 
    f.set_size_inches(10, 10)
    k = 0

    for i in range(height):
        for j in range(width):
            ax[i,j].imshow(images[k].reshape(28, 28) , cmap = "gray")
            k += 1
        plt.tight_layout()
    plt.show()


def show_accuracy_graph(model, history):
    epochs = [i for i in range(EPOCHS)]
    fig , ax = plt.subplots(1,2)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    fig.set_size_inches(16,9)

    ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
    ax[0].plot(epochs , val_acc , 'ro-' , label = 'Testing Accuracy')
    ax[0].set_title('Training & Validation Accuracy')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")

    ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
    ax[1].plot(epochs , val_loss , 'r-o' , label = 'Testing Loss')
    ax[1].set_title('Testing Accuracy & Loss')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    plt.show()


def show_confusion_matrix(y, predictions):
    cm = confusion_matrix(y, predictions)
    cm = pd.DataFrame(cm , index = [i for i in range(25) if i != 9] , columns = [i for i in range(25) if i != 9])
    plt.figure(figsize = (15,15))
    sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='')


def show_predicted_to_correct_classes(x_test, y, predictions, count=6):
    correct = np.nonzero(predictions == y)[0]

    i = 0
    for c in correct[:count]:
        plt.subplot(3,2,i+1)
        plt.imshow(x_test[c].reshape(28,28), cmap="gray", interpolation='none')
        plt.title("Predicted {}/Actual {}".format(CASES[predictions[c]], CASES[y[c]]))
        plt.tight_layout()
        i += 1
    plt.show()
    

def save_model_history(model, history, model_path="model.model",
                       history_path="history.history"):
    # model
    model.save(model_path)

    # history
    with open(history_path, "wb") as f:
        pickle.dump(history, f)


def read_model_history(model_path="model.model",
                       history_path="history.history"):
    try:
        # model
        model = keras.models.load_model(model_path)

        # history
        with open(history_path, "rb") as f:
            history = pickle.load(f)

        return model, history
    except FileNotFoundError:
        return None


def main():
    # preprocessing
    data = list(train_test_split())

    # extracts y
    y = data[4]
    del data[4]

    data = preprocessing(*data)
    extra_data = augment_data(data[0])

    # plot_images(data[0][:10])
    load_from = (MODEL_PATH, HISTORY_PATH)
    # load_from = None
    
    # the actual training
    model, predictions, history, accuracy = train_cnn(*data, extra_data,
                                                      load_from=load_from)

    # saves model/history
    save_model_history(model, history, model_path=MODEL_PATH,
                       history_path=HISTORY_PATH)

    # shows graphs
    # show_accuracy_graph(model, history)
    # show_confusion_matrix(y, predictions)
    # show_predicted_to_correct_classes(data[2], y, predictions)


class RoverCNN:
    CASES = {index:letter.upper() for index,letter in enumerate(string.ascii_lowercase)}


    def __init__(self,
                 epochs = 4,
                 model_path = "piccolo/model.pickle",
                 history_path = "piccolo/history.pickle",
            ):
        # config vals
        self.epochs = epochs
        self.model_path = model_path
        self.history_path = history_path

        # gets train/test data
        self.x_train, self.y_train, self.x_test, self.y_test, self.y = train_test_split()
        # preprocesses it
        self.x_train, self.y_train, self.x_test, self.y_test = preprocessing(*self.data)

        # # data augmentation
        self.extra_data = augment_data(self.x_train)
        # plt.imshow(self.x_train[0], interpolation='nearest')
        # plt.show()

        self.get_model()


    @property
    def data(self):
        return [self.x_train, self.y_train, self.x_test, self.y_test]

    def get_model(self):
        self.model, self.predictions, self.history, self.accuracy = \
            train_cnn(*self.data, self.extra_data, load_from=(self.model_path,
                                                        self.history_path))

    def save(self):
        save_model_history(self.model, self.history,
                           model_path=self.model_path,
                           history_path=self.history_path)

    def predict(self, image):
        print("PREDICTOR", image.shape, len(image))
        processed_image = image / 255
        processed_image = processed_image.reshape(-1,28,28,1)
        # processed_image = processed_image.reshape(1920,480,1)
        results = self.model.predict(processed_image)[0]
        return CASES[list(results).index(max(results))]

    def show_graphs(self):
        show_accuracy_graph(self.model, self.history)


if __name__ == "__main__":
    main()
