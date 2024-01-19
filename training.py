from WordNet import WordNet
import numpy as np
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import argparse
import cv2
import glob
from tensorflow.data import Dataset
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import pickle
import os


np.random.seed(1337)
tf.random.set_seed(1337)
SRC_DIRECTORY = "training_data_playable_2532_augmented"
checkpoint_filepath = '/Users/arvid/Documents/ML_projects/WordFraud/playableletterlassifier'
output_dir = "labels"
BATCH_SIZE = 64
def create_data(no_classes):
    images = []
    labels = []
    for file in glob.glob(SRC_DIRECTORY+"/*.jpg", recursive=True):
        if "empty" in file:
            continue
        img = cv2.imread(file)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255.0
        filename = file.split('/')[1]
        label = filename[0]
        #label = 1 if "SC" in filename else 0
        images.append(img_gray)
        labels.append(label)
    inputs_train = np.array(images)
    inputs_train = inputs_train.reshape(inputs_train.shape[0], 78, 78, 1)
    targets_train = np.array(labels)
    lb = LabelBinarizer()
    targets_train = lb.fit_transform(targets_train)
    #targets_train = np.hstack((targets_train, 1 - targets_train))
    targets_train = targets_train.reshape(targets_train.shape[0], -1)
    ds = Dataset.from_tensor_slices((inputs_train,targets_train))
    no_samples = ds.cardinality().numpy()
    ds = ds.shuffle(no_samples)
    train_ds = ds.take(int(no_samples*0.7))
    val_ds = ds.skip(int(no_samples*0.3))
    train_ds = train_ds.batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)
    with open(os.path.join('playableLetterTokenizer'), 'wb') as f:
        pickle.dump(lb, f)
    return train_ds, val_ds

no_classes = 28
W = WordNet(78, 78, 1, no_classes, no_filters=77, kernel_size=4, no_convlayers=3, pool_size=3, no_fclayers=3, fc_size=512)
model = W.build_model()
train_ds, val_ds = create_data(no_classes)


#{'NUM_FILTERS': 77, 'KERNEL_SIZE': 4, 'NUM_CONVLAYERS': 3, 'POOL_SIZE': 3, 'NUM_FCCLAYERS': 2, 'FC_SIZE': 512}
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
metrics = tf.keras.metrics.CategoricalAccuracy()
model.compile(optimizer=optimizer, metrics=[metrics], loss=loss_fn, run_eagerly=False)
model.fit(
    train_ds,
    validation_data=val_ds,
    batch_size = BATCH_SIZE,
    epochs=100,
    callbacks=[TensorBoard(),
    ModelCheckpoint(checkpoint_filepath, save_best_only = True),
    ]
)

testImg = cv2.imread('training_data/A_unmarked.jpg')
testImg = cv2.cvtColor(testImg, cv2.COLOR_BGR2GRAY)/255.0
testImg = testImg.reshape(1, 78, 78, 1)
print(model(testImg))
testImg = cv2.imread('training_data/H_marked.jpg')
testImg = cv2.cvtColor(testImg, cv2.COLOR_BGR2GRAY)/255.0
testImg = testImg.reshape(1, 78, 78, 1)
print(model(testImg))