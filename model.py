import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D, BatchNormalization, ReLU, Activation, Conv2DTranspose
from tensorflow.keras.optimizers import Adam

import glob
import cv2

# global width
# global height
# global channels
width = 256
height = 256
channels = 3

def InputBlock(channels, input):
    conv = Conv2D(channels, 3, data_format='channels_last', padding = 'same')(input)
    conv = BatchNormalization(axis=3)(conv)
    conv = ReLU()(conv)
    conv = Conv2D(channels, 3, data_format='channels_last', padding = 'same')(conv)
    conv = BatchNormalization(axis=3)(conv)
    output = ReLU()(conv)
    return output

def EncodeBlock(channels, input):
    conv = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(input)
    conv = Conv2D(channels, 3, data_format='channels_last', padding = 'same')(conv)
    conv = BatchNormalization(axis=3)(conv)
    conv = ReLU()(conv)
    conv = Conv2D(channels, 3, data_format='channels_last', padding = 'same')(conv)
    conv = BatchNormalization(axis=3)(conv)
    output = ReLU()(conv)
    return output

def DecodeBlock(channels, input, front):
    conv = Conv2DTranspose(channels/2, 3, (2,2), data_format='channels_last', padding='same')(input)
    conv = BatchNormalization(axis=3)(conv)
    conv = ReLU()(conv)
    # conv = Conv2D(channels/2, 3, data_format='channels_last', padding = 'same')(conv)
    # conv = BatchNormalization(axis=3)(conv)
    # conv = ReLU()(conv)
    conv = concatenate([front,conv], axis=3)
    conv = Conv2D(channels/2, 3, data_format='channels_last', padding = 'same')(conv)
    conv = BatchNormalization(axis=3)(conv)
    conv = ReLU()(conv)
    conv = Conv2D(channels/2, 2, data_format='channels_last', padding = 'same')(conv)
    conv = BatchNormalization(axis=3)(conv)
    output = ReLU()(conv)
    return output

def OutputBlock(input):
    output = Conv2D(1, 1, data_format='channels_last', activation = 'sigmoid')(input)
    return output

def unet(input_size = (width,height,channels)):
    inputs = Input(input_size)
    inputBlock = InputBlock(64, inputs) 
    encodeBlock_1 = EncodeBlock(128,inputBlock)
    encodeBlock_2 = EncodeBlock(256,encodeBlock_1)
    encodeBlock_3 = EncodeBlock(512,encodeBlock_2)
    encodeBlock_4 = EncodeBlock(1024,encodeBlock_3)

    decodeBlock_4 = DecodeBlock(1024,encodeBlock_4, encodeBlock_3)
    decodeBlock_3 = DecodeBlock(512,decodeBlock_4, encodeBlock_2)
    decodeBlock_2 = DecodeBlock(256,decodeBlock_3, encodeBlock_1)
    decodeBlock_1 = DecodeBlock(128,decodeBlock_2, inputBlock)

    outputs = OutputBlock(decodeBlock_1)

    model = Model(inputs = inputs, outputs = [outputs])

    # model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    model.summary()

    return model

def aug_img(image,label):
    img = tf.concat([image,label],axis=-1)
    img = tf.image.resize(img,(256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img = tf.image.random_crop(img,size=[256,256,4])
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        # mask = tf.image.flip_left_right(mask)
        
    if tf.random.uniform(()) > 0.5:
        img = tf.image.adjust_brightness(img,0.5)
        # mask = tf.image.adjust_brightness(mask,0.5)
    
    img = tf.cast(img, tf.float32) / 255.0
    return img[:,:,:3],img[:,:,3:]

def load_data(image, label):
    image = tf.io.read_file(image)
    image = tf.image.decode_png(image, channels=3)
    # image = tf.cast(image, tf.float32)
    
    label = tf.io.read_file(label)
    label = tf.image.decode_png(label, channels=1)

    image, label = aug_img(image, label)

    # label = tf.cast(label, tf.float32) / 255.0
    # one = tf.ones_like(label)
    # zero = tf.zeros_like(label)
    # label = tf.where(label > 0, one, zero)

    # if tf.random.uniform(()) > 0.5:
    #     image = tf.image.flip_left_right(image)
    #     label = tf.image.flip_left_right(label)

    return image, label

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

if __name__ == '__main__':
    epochs = 500
    batch_size = 2

    root = 'data/train/aug/'
    image_list = glob.glob('{}image*'.format(root))
    label_list = list(map(lambda x:x.replace('image', 'mask'), image_list))

    dataset = tf.data.Dataset.from_tensor_slices((image_list, label_list))
    dataset = dataset.repeat(2)
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(len(image_list)*2)
    

    # for d in dataset:
    #     image = d[0].numpy()[0]
    #     label = d[1].numpy()[0]
    #     cv2.imshow('image', image)
    #     cv2.imshow('label', label)
    #     cv2.waitKey()
    # cv2.destroyAllWindows()

    trainset = dataset.take(int(len(image_list)*0.7))
    trainset = trainset.batch(batch_size)
    trainset = trainset.prefetch(tf.data.experimental.AUTOTUNE)

    testset = dataset.skip(int(len(image_list)*0.7) + 1)
    testset = testset.batch(batch_size)

    log = glob.glob('log/*.h5')
    epoch = len(log)
    if epoch > 0:
        model = tf.keras.models.load_model('log/unet_model_{}.h5'.format(epoch))
    else:
        model = unet()

    loss_object = tf.keras.losses.BinaryCrossentropy()

    optimizer = tf.keras.optimizers.Adam(lr = 1e-4)
    # optimizer = tf.keras.optimizers.SGD(lr = 1e-4)
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')


    for e in range(epoch, epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in trainset:
            train_step(images, labels)

        for test_images, test_labels in testset:
            test_step(test_images, test_labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print (template.format(e+1, train_loss.result(), train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100))
        model.save('log/unet_model_{}.h5'.format(e+1))

    model.save('unet_model.h5')