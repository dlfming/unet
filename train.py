import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D, BatchNormalization, ReLU, Activation, Conv2DTranspose

import glob
import os
import cv2.cv2 as cv2  

# 图像的宽
width = 256 
# 图像的高
height = 256
# 图像的通道数
channels = 3
# 数据增强倍率，2表示将数据扩大两倍，比如原始数据100张，增强到200张
augratio = 2
# 训练次数
epochs = 500
# 小批量训练时，每个批次的图像个数
batch_size = 8
# 语义分割的类别数量
num_class = 2
# 选择gpu显卡的序号，0表示第一张显卡
os.environ['CUDA_VISIBLE_DEVICES']='0'

def InputBlock(channels, input):
    conv = Conv2D(channels, 3, data_format='channels_last', padding = 'same')(input)
    conv = BatchNormalization()(conv)
    conv = ReLU()(conv)
    conv = Conv2D(channels, 3, data_format='channels_last', padding = 'same')(conv)
    conv = BatchNormalization()(conv)
    output = ReLU()(conv)
    return output

def EncodeBlock(channels, input):
    conv = MaxPooling2D(pool_size=(2, 2), data_format='channels_last', padding='same')(input)
    conv = Conv2D(channels, 3, data_format='channels_last', padding = 'same')(conv)
    conv = BatchNormalization()(conv)
    conv = ReLU()(conv)
    conv = Conv2D(channels, 3, data_format='channels_last', padding = 'same')(conv)
    conv = BatchNormalization()(conv)
    output = ReLU()(conv)
    return output

def DecodeBlock(channels, input, front):
    conv = Conv2DTranspose(channels//2, 2, (2,2), data_format='channels_last', padding='same')(input)
    conv = BatchNormalization()(conv)
    conv = ReLU()(conv)
    conv = concatenate([conv, front], axis=3)
    conv = Conv2D(channels//2, 3, data_format='channels_last', padding = 'same')(conv)
    conv = BatchNormalization()(conv)
    conv = ReLU()(conv)
    conv = Conv2D(channels//2, 3, data_format='channels_last', padding = 'same')(conv)
    conv = BatchNormalization()(conv)
    output = ReLU()(conv)
    return output

def OutputBlock(input):
    output = Conv2D(num_class, 1, data_format='channels_last', activation = 'softmax')(input)
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

# 数据增强
def aug_img(image,label):
    img = tf.concat([image,label],axis=-1)

    # 随机左右翻转
    if tf.random.uniform(()) > 0.5:
        img = tf.image.random_flip_left_right(img)
    # 随机上下翻转
    if tf.random.uniform(()) > 0.5:
        img = tf.image.random_flip_up_down(img)
    # if tf.random.uniform(()) > 0.5:
    #     seed = [1,2,3,4]
    #     k = random.choice(seed)
    #     img = tf.image.rot90(img, k)

    return img[:,:,:channels],img[:,:,channels:]

def load_train_data(image, label):
    image = tf.io.read_file(image)
    image = tf.image.decode_png(image, channels=channels)
    
    label = tf.io.read_file(label)
    label = tf.image.decode_png(label, channels=1)

    image, label = aug_img(image, label)

    image = tf.image.resize(image,(height, width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    label = tf.image.resize(label,(height, width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    image = tf.cast(image, tf.float32) / 127.5 - 1
    label = tf.cast(label, tf.int32)

    return image, label

def load_test_data(image, label):
    image = tf.io.read_file(image)
    image = tf.image.decode_png(image, channels=channels)
    
    label = tf.io.read_file(label)
    label = tf.image.decode_png(label, channels=1)

    image = tf.image.resize(image,(height, width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    label = tf.image.resize(label,(height, width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    image = tf.cast(image, tf.float32) / 127.5 - 1
    label = tf.cast(label, tf.int32)

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
    # 数据所在根目录
    root = 'E:/Defect detection/Data Annotation/Cityspaces/images/train'
    # 图像路径数组
    image_list = glob.glob('{}/*/*.png'.format(root))
    # 标签路径数组
    label_list = list(map(lambda x:x.replace('images', 'gtFine').replace('leftImg8bit', 'gtFine_labelIds'), image_list))
    
    dataset = tf.data.Dataset.from_tensor_slices((image_list, label_list))
    dataset = dataset.repeat(augratio)
    dataset = dataset.shuffle(dataset.__len__().numpy())

    trainset = dataset.take(int(dataset.__len__().numpy()*0.7))
    trainset = trainset.map(load_train_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    trainset = trainset.cache().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    testset = dataset.skip(int(dataset.__len__().numpy()*0.7))
    testset = testset.map(load_test_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    testset = testset.cache().batch(batch_size)

    # for d in trainset.take(1):
    #     image = d[0].numpy()[0]
    #     label = d[1].numpy()[0]
    #     cv2.imshow('image', image)
    #     cv2.imshow('label', label)
    #     cv2.waitKey()
    # cv2.destroyAllWindows()

    if not os.path.exists('log'): os.mkdir('log')
    log = glob.glob('log/*.h5')
    epoch = len(log)
    if epoch > 0:
        model = tf.keras.models.load_model('log/unet_model_{}.h5'.format(epoch))
    else:
        model = unet()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    optimizer = tf.keras.optimizers.Adam(lr = 1e-3, decay=0.01)
    # optimizer = tf.keras.optimizers.SGD(lr = 1e-4)
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


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