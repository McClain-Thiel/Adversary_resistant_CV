import tensorflow as tf
import pandas as pd

class DataLoader(object):
    """
    TODO:
        I think the best way to train this, if possible, would be to have this class
        generate the perturbations dynamically during training and just feed them
        in with the real images. Not exactly sure how to do that but it would reduce
        data overhead by about 10x.

        Possible ideas:
            1.) wrapper generator that takes only a few images and fills the rest of
            the batch with perturbed images. Might slow down data movment by a lot.


    Dataloader class with attributes:
            train_gen: a tensorflow generator of the training data
            val_gen: same as above, but with validation data
            val_keys: pd.Dataframe that has the classes of the validation set.
                     I honestly don't know the format of this DF
            adv_gen: generator that spits out a collection of perterbed images
                    in the same form as the normal generator

    Initialize with:
        Dataloader(train_file_path, val_file_path, batch_size, input_shape)

    Example:
        data = Dataloader(train_file_path, val_file_path, batch_size, input_shape)
        train_gen, val_gen, val_keys = data.train_gen, data.val_gen, data.val_keys

    """
    def __init__(self, train_file_path, val_file_path, batch_size = 64, input_shape = (64,64), keys_file='val_annotations.txt'):
        im_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        self.train_gen = im_gen.flow_from_directory(directory=train_file_path,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    target_size=input_shape)
        self.val_gen = im_gen.flow_from_directory(directory=val_file_path,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  target_size=input_shape)
        self.val_keys = pd.read_csv(val_file_path + '/' + keys_file, sep='	', header=None)


    def perterb_img(self, img):
        #https://github.com/tensorflow/models/blob/master/research/adversarial_logit_pairing/adversarial_attack.py
        raise NotImplementedError

    def build_perberbed_batch(self, original_batch):
        raise NotImplementedError

    def adv_gen(self):
        """
        this is the actual generator
        :return: tuple
            0: image data in shape (batch_size, (input_shape), color_channel)
            1: one hot encoded label
        """
        raise NotImplementedError