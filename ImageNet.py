import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

import tempfile
from urllib.request import urlretrieve
import tarfile
import os

import json
import matplotlib.pyplot as plt

import PIL
import numpy as np
import time

####################################
#### for CPU only
config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
tf.logging.set_verbosity(tf.logging.ERROR)
sess = tf.InteractiveSession(config=config)
##############################################
########for GPU
#tf.logging.set_verbosity(tf.logging.ERROR)
#sess = tf.InteractiveSession()
##############################################


image = tf.Variable(tf.zeros((299, 299, 3)))
def inception(image, reuse):
    preprocessed = tf.multiply(tf.subtract(tf.expand_dims(image, 0), 0.5), 2.0)
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, _ = nets.inception.inception_v3(
            preprocessed, 1001, is_training=False, reuse=reuse)
        logits = logits[:,1:] # ignore background class
        probs = tf.nn.softmax(logits) # probabilities
    return logits, probs

logits, probs = inception(image, reuse=False)



'''
data_dir = tempfile.mkdtemp()
#inception_tarball, _ = urlretrieve(
#    'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz')
inception_tarball = '../inception_v3_2016_08_28.tar.gz'
tarfile.open(inception_tarball, 'r:gz').extractall(data_dir)
'''

data_dir = '../Downloads'
restore_vars = [
    var for var in tf.global_variables()
    if var.name.startswith('InceptionV3/')
]
saver = tf.train.Saver(restore_vars)
saver.restore(sess, os.path.join(data_dir, 'inception_v3.ckpt'))


imagenet_json = '../imagenet.json'
with open(imagenet_json) as f:
    imagenet_labels = json.load(f)


def classify(img, correct_class=None, target_class=None, pars=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    fig.sca(ax1)
    p = sess.run(probs, feed_dict={image: img})[0]
    ax1.imshow(img)
    fig.sca(ax1)

    topk = list(p.argsort()[-10:][::-1])
    topprobs = p[topk]
    barlist = ax2.bar(range(10), topprobs)
    if target_class in topk:
        barlist[topk.index(target_class)].set_color('r')
    if correct_class in topk:
        barlist[topk.index(correct_class)].set_color('g')



    plt.sca(ax2)
    plt.ylim([0, 1.1])
    plt.xticks(range(10),
               [imagenet_labels[i][:15] for i in topk],
               rotation='vertical')


    fig.subplots_adjust(bottom=0.2)
    plt.show()
    for rect in barlist:
        height = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,'%0.3f' % float(height), ha='center', va='bottom')
    if pars:
        plt.title('epsilon: '+str(pars[0])+', learning rate: '+str(pars[1])+', no. loops: '+str(pars[2]))


def show_result(img,adv):
    plt.figure(3)
    plt.subplot(121)
    plt.title('original image')
    plt.imshow(img)
    plt.subplot(122)
    plt.title('adversarial sample')
    plt.imshow(adv)


###################
#import images for classifying
###################


def save_image(img,Img_path,opt):
    reform_img = PIL.Image.fromarray((img*255.0).astype('uint8'))
    path, filename = os.path.split(Img_path)
    name, extension = filename.split(".")
    reform_img.save(path+'/'+name+'_'+opt,'JPEG')

def resize_image(img,basewidth):
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS) # Image.ANTIALIAS: a high-quality downsampling filter
    return img


img_path = '../street_sign3.jpg'
#img_name=os.path.basename(img_path)
#img_name, file_extension = os.path.splitext(img_name)

#img_class = 281 #cat
#img_class = 673 # computer mouse
img_class = 919 # street sign
img = PIL.Image.open(img_path)
'''
big_dim = max(img.width, img.height)
wide = img.width > img.height
new_w = 299 if not wide else int(img.width * 299 / img.height)
new_h = 299 if wide else int(img.height * 299 / img.width)
img = img.resize((new_w, new_h)).crop((0, 0, 299, 299))
resize_image(img,299)
'''

img = img.resize((299, 299), PIL.Image.ANTIALIAS) # Image.ANTIALIAS: a high-quality downsampling filter
img = (np.asarray(img) / 255.0).astype(np.float32)
#save_image(img,img_path,"resized")

classify(img, correct_class=img_class)


#########################################
#craft the adversarial examples images
#########################################

x = tf.placeholder(tf.float32, (299, 299, 3))
x_hat = image # our trainable adversarial input
assign_op = tf.assign(x_hat, x)

#Gradient descent step

learning_rate = tf.placeholder(tf.float32, ())
y_hat = tf.placeholder(tf.int32, ())

labels = tf.one_hot(y_hat, 1000)
#loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=[labels])
loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=[labels])
optim_step = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(loss, var_list=[x_hat])

###################
#Projection step
###################

epsilon = tf.placeholder(tf.float32, ())

below = x - epsilon
above = x + epsilon
projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
with tf.control_dependencies([projected]):
    project_step = tf.assign(x_hat, projected)

###################
#Execution
###################

demo_epsilon = 8.0 / 255.0  # a really small perturbation
#demo_epsilon = 0.3/255.0
#demo_lr = 1e-1  # learning rate
demo_lr = 2e-1
demo_steps = 10  #iterative
paras = [demo_epsilon,demo_lr,demo_steps]
#demo_target = 924  # "guacamole"
#demo_target = 673  # "computer mouse"
demo_target = 571  # "gas pump"
# initialization step
sess.run(assign_op, feed_dict={x: img})

# projected gradient descent
startTime = time.time()
for i in range(demo_steps):
    # gradient descent step
    _, loss_value = sess.run(
        [optim_step, loss],
        feed_dict={learning_rate: demo_lr, y_hat: demo_target})
    # project step
    sess.run(project_step, feed_dict={x: img, epsilon: demo_epsilon})
    if (i + 1) % 2 == 0:
        print('step %d, loss=%g' % (i + 1, loss_value))

elapsedTime = time.time() - startTime
print("elapsed Time: ", elapsedTime)
adv = x_hat.eval()  # retrieve the adversarial example

classify(adv, correct_class=img_class, target_class=demo_target,pars=paras)

show_result(img,adv)

save_image(adv,img_path,"adv2")
##############################
#Robust adversarial examples
##############################

ex_angle = np.pi/8

angle = tf.placeholder(tf.float32, ())
rotated_image = tf.contrib.image.rotate(image, angle)
rotated_example = rotated_image.eval(feed_dict={image: adv, angle: ex_angle})
classify(rotated_example, correct_class=img_class, target_class=demo_target)

num_samples = 10
average_loss = 0
for i in range(num_samples):
    rotated = tf.contrib.image.rotate(
        image, tf.random_uniform((), minval=-np.pi/4, maxval=np.pi/4))
    rotated_logits, _ = inception(rotated, reuse=True)
    average_loss += tf.nn.softmax_cross_entropy_with_logits(
        logits=rotated_logits, labels=labels) / num_samples

optim_step = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(average_loss, var_list=[x_hat])

demo_epsilon = 8.0 / 255.0  # still a pretty small perturbation
demo_lr = 2e-1
demo_steps = 10
paras = [demo_epsilon,demo_lr,demo_steps]
#demo_target = 924  # "guacamole"
#demo_target = 510  # "container ship"
demo_target = 571  # "gas pump"
# initialization step
sess.run(assign_op, feed_dict={x: img})

# projected gradient descent
startTime = time.time()
for i in range(demo_steps):

    # gradient descent step
    _, loss_value = sess.run(
        [optim_step, average_loss],
        feed_dict={learning_rate: demo_lr, y_hat: demo_target})
    # project step
    sess.run(project_step, feed_dict={x: img, epsilon: demo_epsilon})
    if (i + 1) % 50 == 0:
        print('step %d, loss=%g' % (i + 1, loss_value))

elapsedTime = time.time() - startTime
print("elapsed Time: ", elapsedTime)
adv_robust = x_hat.eval()  # retrieve the adversarial example

rotated_example = rotated_image.eval(feed_dict={image: adv_robust, angle: ex_angle})
classify(rotated_example, correct_class=img_class, target_class=demo_target,pars=paras)

###################
#Evaluation
###################

thetas = np.linspace(-np.pi / 4, np.pi / 4, 301)
p_naive = []
p_robust = []
for theta in thetas:
    rotated = rotated_image.eval(feed_dict={image: adv_robust, angle: theta})
    p_robust.append(probs.eval(feed_dict={image: rotated})[0][demo_target])

    rotated = rotated_image.eval(feed_dict={image: adv, angle: theta})
    p_naive.append(probs.eval(feed_dict={image: rotated})[0][demo_target])

robust_line, = plt.plot(thetas, p_robust, color='b', linewidth=2, label='robust')
naive_line, = plt.plot(thetas, p_naive, color='r', linewidth=2, label='naive')
plt.ylim([0, 1.05])
plt.xlabel('rotation angle')
plt.ylabel('target class probability')
plt.legend(handles=[robust_line, naive_line], loc='lower right')
plt.show()

