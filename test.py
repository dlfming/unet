import tensorflow as tf
import cv2
import numpy as np

def load_data(image):
    image = tf.io.read_file(image)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, (256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, 0)

    return image

if __name__ == '__main__':
    print(tf.__version__)
    print(tf.test.is_gpu_available())
    model = tf.keras.models.load_model("log/unet_model_20.h5")

    test = 'data/train/image/0.png'
    truth = 'data/train/label/0.png'

    test = load_data(test)
    # print(test.shape)
    # print(test.numpy()[0])
    # print(np.unique(test.numpy()[0]))

    # test = cv2.imread(test, -1)
    # print(test.shape)
    # print(test)
    # print(np.unique(test))

    truth = cv2.imread(truth)
        
    cv2.imshow('test', test.numpy()[0])
    cv2.imshow('truth', truth)

    predict = model(test)
    # print(predict.shape)
    result = predict.numpy()[0]
    print(result.dtype)
    result = cv2.convertScaleAbs(result)
    result = result * 255.0
    print(np.unique(result))

    cv2.imshow('predict', result)
    cv2.imwrite('predict.png', result)

    cv2.waitKey()
    cv2.destroyAllWindows()

