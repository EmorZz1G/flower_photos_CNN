import tensorflow as tf
import numpy as np
import cv2

path1 = r"D:/Users/Administrator/PycharmProjects/AI/flower_photos_CNN/pic/0.jpg"
path2 = "./pic/1.jpg"
path3 = "./pic/2.jpg"
path4 = "./pic/3.jpg"
path5 = "./pic/4.jpg"
w = 100
h = 100


def img(path):
    print("fk", path)
    im = cv2.imread(path)
    _img = cv2.resize(im, (w, h))
    return np.asarray(_img)


Map = {0: "雏菊(daisy)", 1: "蒲公英(dandelion)", 2: "玫瑰(rose)", 3: "向日葵(sunflower)", 4: "郁金香(tulips)"}


def guess(path):
    # imgs = [img(path)]
    imgs = [img(path)]
    # imgs.append(img(path1))
    # imgs.append(img(path2))
    # imgs.append(img(path3))
    # imgs.append(img(path4))
    # imgs.append(img(path5))

    sess = tf.Session()
    saver = tf.train.import_meta_graph("./model/flower/flower1.meta")
    saver.restore(sess, "./model/flower/flower1")

    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name("x:0")
    logits = graph.get_tensor_by_name("logits_eval:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")

    feed_dict = {x: imgs, keep_prob: 1.0}
    classify_result = sess.run(logits, feed_dict=feed_dict)
    res = []
    for i in range(len(classify_result)):
        j = np.argmax(classify_result[i], 0)
        res.append(Map[int(j)])
        print("预测 : ", Map[int(j)])
    return res
