import os
import pandas as pd
import numpy as np
import cv2
from keras.layers import GlobalAveragePooling2D
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG19, InceptionV3, ResNet50, MobileNetV2, DenseNet121


########################################################################################################################

def load_sixray_data(data_dir, dataset_type):
    if dataset_type not in ['train', 'valid', 'test']:
        raise ValueError("Invalid dataset_type. Must be either 'train', 'valid' or 'test'")

    # Load annotations
    annotations_file = os.path.join(data_dir, dataset_type, '_annotations.csv')
    annotations = pd.read_csv(annotations_file)

    # Load images and preprocess them
    images = []
    for filename in annotations['filename']:
        image_file = os.path.join(data_dir, dataset_type, filename)
        image = cv2.imread(image_file)
        image = cv2.resize(image, (256, 256))
        image = img_to_array(image)
        images.append(image)

    images = np.array(images, dtype="float32") / 255.0

    # Binarize annotations
    mlb = MultiLabelBinarizer()
    annotations = mlb.fit_transform(annotations["class"].str.split(","))
    return images, annotations


########################################################################################################################
# 12
def build_cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(256, 256, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(5, activation="sigmoid"))
    return model


def run_cnn_model(loss, optimizer, metrics, epochs, batch_size, name):
    cnn_model = build_cnn_model()
    cnn_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    history = cnn_model.fit(train_images, train_annotations, epochs=epochs, batch_size=batch_size,
                            validation_data=(valid_images, valid_annotations))
    cnn_model.save(f'{name}.h5')
    return history
########################################################################################################################
# 19+4
def build_VGG19_model():
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='sigmoid'))
    return model

def run_VGG19_model(loss, optimizer, metrics, epochs, batch_size):
    vgg19_model = build_VGG19_model()
    vgg19_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    history = vgg19_model.fit(train_images, train_annotations, epochs=epochs, batch_size=batch_size,
                              validation_data=(valid_images, valid_annotations))
    vgg19_model.save('vgg19_model.h5')
    return history
########################################################################################################################
#42+4
def build_inceptionv3_model():
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='sigmoid'))
    return model

def run_inceptionv3_model(loss, optimizer, metrics, epochs, batch_size):
    inceptionv3_model = build_inceptionv3_model()
    inceptionv3_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    history = inceptionv3_model.fit(train_images, train_annotations, epochs=epochs, batch_size=batch_size,
                                    validation_data=(valid_images, valid_annotations))
    inceptionv3_model.save('inceptionv3_model.h5')
    return history
########################################################################################################################
# 50+4
def build_resnet_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='sigmoid'))
    return model

def run_resnet_model(loss, optimizer, metrics, epochs, batch_size):
    resnet_model = build_resnet_model()
    resnet_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    history = resnet_model.fit(train_images, train_annotations, epochs=epochs, batch_size=batch_size,
                               validation_data=(valid_images, valid_annotations))
    resnet_model.save('resnet_model.h5')
    return history
########################################################################################################################
#88+4
def build_mobilenetv2_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='sigmoid'))
    return model

def run_mobilenetv2_model(loss, optimizer, metrics, epochs, batch_size):
    mobilenetv2_model = build_mobilenetv2_model()
    mobilenetv2_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    history = mobilenetv2_model.fit(train_images, train_annotations, epochs=epochs, batch_size=batch_size,
                                    validation_data=(valid_images, valid_annotations))
    mobilenetv2_model.save('mobilenetv2_model.h5')
    return history
########################################################################################################################
#121+4
def build_densenet121_model():
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='sigmoid'))
    return model

def run_densenet121_model(loss, optimizer, metrics, epochs, batch_size):
    densenet121_model = build_densenet121_model()
    densenet121_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    history = densenet121_model.fit(train_images, train_annotations, epochs=epochs, batch_size=batch_size,
                                    validation_data=(valid_images, valid_annotations))
    densenet121_model.save('densenet121_model.h5')
    return history
########################################################################################################################
def plot_for_training_validation_accuracy_loss(history, name):
    # accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(name + 'accuracy.png')
    plt.show()
    # loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(name + 'loss.png')
    plt.show()
    return 'Done'
########################################################################################################################
if __name__ == "__main__":
    train_images, train_annotations = load_sixray_data('Path_to_dataset', 'train')
    valid_images, valid_annotations = load_sixray_data('Path_to_dataset', 'valid')
    test_images, test_annotations = load_sixray_data('Path_to_dataset', 'test')
########################################################################################################################
    history_cnn_model_1 = run_cnn_model(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"],
                                        epochs=2, batch_size=16, name="cnn_model_1")
    history_cnn_model_2 = run_cnn_model(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"],
                                         epochs=50, batch_size=32, name="cnn_model_2")
    history_cnn_model_3 = run_cnn_model(loss="binary_crossentropy", optimizer="Adagrad", metrics=["accuracy"],
                                         epochs=50, batch_size=16, name="cnn_model_3")
    history_cnn_model_4 = run_cnn_model(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"],
                                         epochs=50, batch_size=64, name="cnn_model_4")
########################################################################################################################
    history_vgg19_model = run_VGG19_model(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"],
                                          epochs=1, batch_size=32)
    vgg19_plots = plot_for_training_validation_accuracy_loss(history_vgg19_model, 'vgg19')
########################################################################################################################
    history_inceptionv3_model = run_inceptionv3_model(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"],
                                          epochs=1, batch_size=32)
    inceptionv3_plots = plot_for_training_validation_accuracy_loss(history_inceptionv3_model, 'inceptionv3')
########################################################################################################################
    history_resnet_model = run_resnet_model(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"],
                                            epochs=1, batch_size=32)
    resnet50_plots = plot_for_training_validation_accuracy_loss(history_resnet_model,'resnet50')
########################################################################################################################
    history_mobilenetv2_model = run_mobilenetv2_model(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"],
                                          epochs=1, batch_size=32)
    mobilenetv2_plots = plot_for_training_validation_accuracy_loss(history_mobilenetv2_model, 'mobilenetv2')
########################################################################################################################
    history_densenet_model = run_densenet121_model(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"],
                                            epochs=1, batch_size=32)
    densenet121_plots = plot_for_training_validation_accuracy_loss(history_densenet_model, 'densenet121')
########################################################################################################################






