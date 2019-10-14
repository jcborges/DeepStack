from deepstack import Member
from deepstack import DirichletEnsemble
from deepstack import StackEnsemble
from keras.models import Sequential
import random
from keras.utils import to_categorical
from keras.datasets import cifar10
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from sklearn.ensemble import RandomForestRegressor
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization


def load_sample_cifar_dataset(trainsample=5000, testsample=500):
    """
    Loads a sample of cifar dataset. For training it creates a random sampling.
    For validation and testing it creates a fixed sample.
    The rationale is to train algorithms on different training sets but validate and test on the same dataset in order
    to guarantee comparability.
    Args:
        trainsample: site of training set
        testsample: size of validation / test set

    Returns: x,y datasets for train, test and validation
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    trainindex = random.sample(list(range(x_train.shape[0])), trainsample)
    return x_train[trainindex, :, :, :], y_train[trainindex, :], x_test[0:testsample, :, :, :], y_test[0:testsample, :], x_test[testsample:testsample * 2, :, :, :], y_test[testsample:testsample * 2, :]


def create_random_cnn(input_shape):
    """
    Creates a CNN, based on random layer size.
    Idea is to generate similar CNN models per function call
    Args:
        input_shape: the input_shape of the model

    Returns: a keras CNN model
    """
    weight_decay = 1e-4
    num_classes = 10
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                     input_shape=input_shape))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(random.randint(16, 64), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(random.randint(0, 5) * 0.1))

    model.add(Conv2D(random.randint(16, 64), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(random.randint(16, 64), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(random.randint(0, 5) * 0.1))

    model.add(Conv2D(random.randint(64, 128), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(random.randint(64, 128), (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(random.randint(0, 5) * 0.1))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    return model


def get_random_cifar_model(batch_size=32, epochs=100):
    """
    Creates and fits a (random) CNN on the CIFAR10 dataset.
    Args:
        batch_size: the batch size for training the CNN model
        epochs: epochs to train the model

    Returns: fitted CNN model for the Cifar10 dataset, validation batches and test batches
    """
    opt_rms = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
    datagen = ImageDataGenerator(rotation_range=90,
                                 width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    x_train, y_train, x_val, y_val, x_test, y_test = load_sample_cifar_dataset()
    model = create_random_cnn(input_shape=x_train.shape[1:])
    datagen.fit(x_train)
    model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])
    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size, epochs=epochs,
                        verbose=0, validation_data=(x_val, y_val), callbacks=[es_callback])
    validation_batches = datagen.flow(x_val, y_val, batch_size=batch_size)
    test_batches = datagen.flow(x_test, y_test, batch_size=batch_size)
    return model, validation_batches, test_batches


def cifar10_example(nmembers=4):
    """
    Runs 2 DeepStack Ensemble Models for the Cifar Dataset
    Args:
        nmembers: amount of ensemble members to be generated

    Returns: an instance of StackEnsemble and DirichletEnsemble for the Cifar10 dataset
    """
    stack = StackEnsemble()
    stack.model = RandomForestRegressor(verbose=1, n_estimators=300 * nmembers,
                                        max_depth=nmembers * 2, n_jobs=4)  # Meta-Learner
    dirichletEnsemble = DirichletEnsemble(N=2000 * nmembers)

    for i in range(nmembers):
        # Creates a Random CNN Keras Model for Cifar10 Dataset
        model, training_batch, validation_batch = get_random_cifar_model()
        """
        Rationale: The Validation and Testing dataset of a base-learner is the Training
        and Validation Dataset of a Meta-Learner. Idea is to avoid validating the meta-learner
        on data that the base-learner has already seen on training
        """
        member = Member(name="model" + str(i + 1), keras_model=model, train_batches=training_batch,
                        val_batches=validation_batch)  # Base-Learners
        stack.add_member(member)  # Adds base-learner to Stack Ensemble
        dirichletEnsemble.add_member(member)  # Adds base-learner to Dirichlet Ensemble
    stack.fit()
    dirichletEnsemble.fit()
    return stack, dirichletEnsemble


# Run Examples
stack, dirichletEnsemble = cifar10_example()

stack.describe()
"""
Possible similar Output:
model1 - AUC: 0.8044
model2 - AUC: 0.8439
model3 - AUC: 0.8218
model4 - AUC: 0.8487
StackEnsemble AUC: 0.8727
"""

dirichletEnsemble.describe()
"""
Possible similar Output:
model1 - Weight: 0.1055 - AUC: 0.8044
model2 - Weight: 0.2882 - AUC: 0.8439
model3 - Weight: 0.2127 - AUC: 0.8218
model4 - Weight: 0.3936 - AUC: 0.8487
DirichletEnsemble AUC: 0.8821
"""
