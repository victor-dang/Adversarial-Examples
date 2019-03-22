import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm
from affine_demo import scaling_image, show_images, translation_image, translation_gray_image
import operator
import time
from six.moves import xrange
from mpl_toolkits.mplot3d import Axes3D
#Read the input data
mnist = input_data.read_data_sets("MNIST-data/", one_hot=True)

#Explore the data
sample_image = mnist.train.next_batch(1)[0]
print("Training image shape", mnist.train.images.shape)
print("Training labels shape", mnist.train.labels.shape)
print("Shape of an image", sample_image.shape)
sample_image = sample_image.reshape([28, 28])
plt.imshow(sample_image)

image_size = 28
labels_size = 10
learning_rate = 0.05
steps_number = 1000
batch_size = 100

# Placeholder is a value that we input when we ask TensorFlow to run a computation.
x = tf.placeholder(tf.float32, shape = [None,784])
y_ = tf.placeholder(tf.float32, shape = [None, 10])

# Functions for weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# Function for bias
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Functions for convolution with stride = 1 and padding = 0
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# Function for max pooling over 2*2 blocks
def max_pooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#https://www.tensorflow.org/get_started/mnist

# Input layer where single input image shape is (1,784)
x_image = tf.reshape(x, [-1,28,28,1])

# Convolution layer 1 - 32 x 5 x 5
# Conv -> BatchNorm -> Relu -> Max_pool
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_conv1 = conv2d(x_image, W_conv1) + b_conv1
y1 = tf.nn.relu(tf.layers.batch_normalization(x_conv1))
x_pool1 = max_pooling_2x2(y1)

# Conv layer 2 - 64 x 5 x 5
# Conv -> BatchNorm -> Relu -> Max_pool
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
x_conv2 = conv2d(x_pool1, W_conv2) + b_conv2
y2 = tf.nn.relu(tf.layers.batch_normalization(x_conv2))
x_pool2 = max_pooling_2x2(y2)

# Flatten
x_flat = tf.reshape(x_pool2, [-1, 7 * 7 * 64])

# Dense fully connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024]) # max pooling reduced image to 7x7
b_fc1 = bias_variable([1024])
x_fc1 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(x_flat, W_fc1) + b_fc1))

# Dropout
keep_prob = tf.placeholder(tf.float32)
x_fc1_drop = tf.nn.dropout(x_fc1, keep_prob)

# Classification layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(x_fc1_drop, W_fc2) + b_fc2

# Probabilities output from model
y = tf.nn.softmax(y_conv)

# Loss and Adam optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_, logits=y_conv))
learning_rate = 1e-3
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# Test accuracy of model
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()

#tf.add_to_collection('x', x)

sess.run(tf.global_variables_initializer())

###########################
# Train model
#### only run this section for the first time, next, it should be restored from Saved_Models

for i in tqdm(range(200)):
    batch = mnist.train.next_batch(100)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("Epoch %d, training accuracy %g"%(i, train_accuracy))

    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.4})


print(train_accuracy)

print((mnist.test.labels[:,2]))

###########################
#### Save model

saver = tf.train.Saver()
saver.save(sess, "Saved_Models/mnist.ckpt")


###########################
### restore model#########
saver = tf.train.Saver()
saver.restore(sess,"Saved_Models/mnist.ckpt")
'''
imported_graph = tf.train.import_meta_graph('Saved_Models/mnist.ckpt.meta')
#saver.restore(sess,tf.train.latest_checkpoint('Saved_Models/'))
imported_graph.restore(sess,"Saved_Models/mnist.ckpt")

#x = tf.get_collection('x')[0]
'''
########################


# Fetch 10 4's images
original_number = 6
number_images = 10
index_mask = np.where(mnist.test.labels[:, original_number])[0]
print(index_mask)
subset_mask = np.random.choice(index_mask, number_images)
print(subset_mask)

# Display the labels of the selected images
original_images = mnist.test.images[subset_mask]
original_labels = mnist.test.labels[subset_mask]
original_labels

# Predict the model on the selected 10 4's samples


prediction = tf.argmax(y,1)
#prediction_val = sess.run(prediction,feed_dict={x: original_images, keep_prob: 1.0})
prediction_val = prediction.eval(feed_dict={x: original_images, keep_prob: 1.0}, session=sess)
print("predictions", prediction_val)
probabilities = y
probabilities_val = probabilities.eval(feed_dict={x: original_images, keep_prob: 1.0}, session=sess)
print ("probabilities\n", np.round(probabilities_val,3))

# Show the original images, correct, predicted label and confidence
def show_image_list(input_image_list,number_images=10):
    for i in range(0, number_images):
        print('Correct label', np.argmax(original_labels[i]))
        print('Predicted label:', prediction_val[i])
        print('Confidence:', np.max(probabilities_val[i]))
        plt.figure(figsize=(2, 2))
        plt.axis('off')
        plt.imshow(input_image_list[i].reshape([28, 28]))
        plt.show()



# Set the target label as 7
target_number = 9
target_labels = np.zeros(original_labels.shape)
target_labels[:, target_number] = 1

target_labels


#perturbation = 0.03 #The amount to wiggle towards the gradient of target class.
#steps = 500

def fgsm_f(input,perturbation = 0.03, steps = 500):
    img_gradient = tf.gradients(cross_entropy, x)[0]
    adversarial_images = input.copy()
    for i in range(0, steps):
        gradient = img_gradient.eval({x: adversarial_images, y_: target_labels, keep_prob: 1.0})
        #Update using value of gradient
        adversarial_images = adversarial_images - perturbation * gradient
        prediction = tf.argmax(y,1)
        prediction_val = prediction.eval(feed_dict={x: adversarial_images, keep_prob: 1.0}, session=sess)
        probabilities = y
        probabilities_val = probabilities.eval(feed_dict={x: adversarial_images, keep_prob: 1.0}, session=sess)
        #print('Confidence 4:\n', np.round(probabilities_val[:, 4],3))
        #print('Confidence 7:\n', np.round(probabilities_val[:, 7],3))
        if(i%50==0):
            print("\n step: %d" % (i + 1))
            print("Prediction results: ", prediction_val)
            print('Probabilities is digit {}: {}\n'.format(original_number,np.round(probabilities_val[:, original_number],3)))
            print('Probabilities is digit {}: {}\n'.format(target_number,np.round(probabilities_val[:, target_number],3)))
    return adversarial_images

#perturbation = 0.02
#steps = 10

def ifgsm_f(input,perturbation = 0.02,steps = 10):
    adversarial_images = input.copy()
    adv_img_hist = list()
    prob_4_hist = list()
    prob_7_hist = list()
    img_gradient = tf.gradients(cross_entropy, x)[0]
    for i in range(0, steps):
        gradient = img_gradient.eval({x: adversarial_images, y_: target_labels, keep_prob: 1.0})
        # Update using sign of gradient and decreasing the step size
        adversarial_images = adversarial_images - perturbation * np.sign(gradient)
        adv_img_hist.append(adversarial_images)
        prediction = tf.argmax(y,1)
        prediction_val = prediction.eval(feed_dict={x: adversarial_images, keep_prob: 1.0}, session=sess)
        print("Prediction results:", prediction_val)
        probabilities = y
        probabilities_val = probabilities.eval(feed_dict = {x: adversarial_images, keep_prob: 1.0}, session=sess)
        #print('Confidence 4:', np.round(probabilities_val[:, 4],3))
        #print('Confidence 7:', np.round(probabilities_val[:, 7],3))
        print('Probabilities is digit {}: {}\n'.format(original_number,np.round(probabilities_val[:, original_number],3)))
        print('Probabilities is digit {}: {}\n'.format(target_number,np.round(probabilities_val[:, target_number],3)))
    prob_4_hist.append(probabilities_val[:, 4])
    prob_7_hist.append(probabilities_val[:, 7])
    return adversarial_images

def pgd_f(input_image,targeted_class,number_images = 10,perturbation = 0.02,steps = 10):
    x_hat = tf.Variable(tf.zeros((1,784))) # our trainable adversarial input
    #x = tf.placeholder(tf.float32, (299, 299, 3))
    assign_op = tf.assign(x_hat, x)
    loss = tf.gradients(cross_entropy, x)[0]
    # Gradient descent step

    learning_rate = tf.placeholder(tf.float32, ())
    #y_hat = tf.placeholder(tf.int32, ())

    #labels = tf.one_hot(y_, 10)
    #loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=[labels])
    #optim_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, var_list=[x_hat])
    optim_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    ###################
    # Projection step
    ###################

    epsilon = tf.placeholder(tf.float32, ())
    below = x - epsilon
    above = x + epsilon
    projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
    with tf.control_dependencies([projected]):
        project_step = tf.assign(x_hat, projected)

    ###################
    # Execution
    ###################

    epsilon_list = (0.001, 0.03, 0.1, 1)
    perturbation = epsilon_list[1]
    input_learningRate = 1e-1  # learning rate
    step_size = 50  # iterative

    # initialization step
    sess.run(assign_op, feed_dict={x: input_image})

    # projected gradient descent
    loss_value_list = np.empty((0))
    startTime = time.time()
    for i in range(step_size):
        # gradient descent step
        _, loss_value = sess.run(
            [optim_step, loss],
            feed_dict={learning_rate: input_learningRate, y_: targeted_class, x:input_image})
        # project step
        sess.run(project_step, feed_dict={x: input_image, epsilon: perturbation})
        if (i + 1) % 2 == 0:
            print('step %d, loss=%g' % (i + 1, loss_value))
        loss_value_list = np.append(loss_value_list, loss_value)
    elapsedTime = time.time() - startTime
    print("elapsed Time: ", elapsedTime)
    adversarial_image = x_hat.eval()  # retrieve the adversarial example
    return adversarial_image

#adv = pgd_f(input_image=original_images[0].reshape(-1,784),targeted_class=target_labels[0].reshape(-1,10),number_images = 1,perturbation = 0.02,steps = 10)

# Generate the image of 10 samples with original_image, delta and adversarial image
def show_ori_adv_images(ori_image_list,adv_image_list, number_images = 10 ):
    f, axarr = plt.subplots(number_images, 3, figsize=(5, 15))
    import matplotlib as mpl

    mpl.rcParams['figure.dpi'] = 250
    for i in range(number_images):
        axarr[i, 0].set_axis_off()
        axarr[i, 1].set_axis_off()
        axarr[i, 2].set_axis_off()
        axarr[i, 0].imshow(ori_image_list[i].reshape([28, 28]))
        axarr[i, 1].imshow((adv_image_list[0][i] - ori_image_list[i]).reshape([28, 28]), cmap = "gray")
        axarr[i, 2].imshow((adv_image_list[0][i]).reshape([28, 28]))
    plt.tight_layout()
    plt.show()


###############################
#### translation##############
###############################
def translation_f(input,input_tx=1,input_ty=1, n_images = 10):
    image = tf.Variable(tf.zeros((28,28)))
    translated_images = np.empty([0,28*28])
    for i in range(n_images):
        input_img = input[i].reshape([28,28])
        translated_image = translation_gray_image(input_img,tx=input_tx,ty=input_ty)
        translated_flatten_image = translated_image.reshape((-1,28*28))
        translated_images = np.append(translated_images,translated_flatten_image, axis=0)
    return translated_images

def translation_probs_list(input,correct_class = None, tx = 1, ty = 1,n_images = 10):
    original_prob_dict = dict()
    target_prob_dict = dict()
    for tx_i in range(0,tx):
        for ty_j in range(0,ty):
            translated_images = np.empty([0, 28 * 28])
            for i in range(n_images):
                input_img = input[i].reshape([28, 28])
                translated_image = translation_gray_image(input_img, tx=tx_i, ty=ty_j)
                translated_flatten_image = translated_image.reshape((-1, 28 * 28))
                translated_images = np.append(translated_images, translated_flatten_image, axis=0)

            prediction = tf.argmax(y, 1)
            # prediction_val = sess.run(prediction,feed_dict={x: original_images, keep_prob: 1.0})
            prediction_val = prediction.eval(feed_dict={x: translated_images, keep_prob: 1.0}, session=sess)
            print("tx= ",tx_i,", ty= ",ty_j)
            print("predictions", prediction_val)
            probabilities = y
            probabilities_val = probabilities.eval(feed_dict={x: translated_images, keep_prob: 1.0},
                                                       session=sess)
            # print ("probabilities\n", np.round(probabilities_val,3))
            print('Probabilities is digit {}: {}\n'.format(original_number, np.round(probabilities_val[:, original_number], 3)))
            print('Probabilities is digit {}: {}\n'.format(target_number, np.round(probabilities_val[:, target_number], 3)))
            for i in range(n_images):
                original_prob_dict[(i, tx_i, ty_j)] = np.round(probabilities_val[i, original_number], 3)
                target_prob_dict[(i,tx_i,ty_j)] = np.round(probabilities_val[i, target_number],3)

    return original_prob_dict,target_prob_dict

adversarial_img = fgsm_f(input = original_images)
translation_ori_probs_dict,translation_tar_probs_dict = translation_probs_list(input=adversarial_img, tx = 4, ty = 4,n_images = 10)

def getKeysByValue(dictOfElements, valueToFind):
    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return  listOfKeys

def plot_3D_dict(input, input_dic, img_name='digit 0', class_label ='digit 0',n_images = 10,fig_saving=0,fig_size=[2093,905],save_path=None):
    for img_index in range(n_images):
        #index_max_x, index_max_y = max(input_dic.items(), key=operator.itemgetter(1))[0]
        #max_z = input_dic[(index_max_x,index_max_y)]

        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.imshow(input[img_index].reshape([28,28]))
        ax1.set_title(img_name)
        ax = fig.add_subplot(122, projection='3d')
        X = list()
        Y = list()
        Z = list()
        for key, value in input_dic.items():
            index, xi, yi = key
            if(index==img_index):
                X.append(xi)
                Y.append(yi)
                Z.append(value)
        x = np.asarray(X)
        y = np.asarray(Y)
        z = np.asarray(Z)
        max_z = max(z)
        keys = getKeysByValue(input_dic,np.float32(max_z))
        index_max_x = keys[0][1]
        index_max_y = keys[0][2]
        ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
        ax.set_xlabel('tx')
        ax.set_ylabel('ty')
        ax.set_zlabel('probability for '+str(class_label))
        ax.set_title('max probility: '+ str(round(max_z,3))+', at ('+str(index_max_x)+','+str(index_max_y)+')')
        plt.show()
        if (fig_saving):
            fig1 = plt.gcf()
            DPI = fig1.get_dpi()
            width, height = fig_size
            fig1.set_size_inches(width / float(DPI), height / float(DPI))
            filename = 'figure' + str(img_index) + '-1.png'
            path = save_path
            fig1.savefig(path + filename, bbox_inches='tight');

save_img_path = "../your_path/"
plot_3D_dict(adversarial_img,translation_ori_probs_dict,fig_saving=1,save_path=save_img_path,n_images = 10)
plot_3D_dict(adversarial_img,translation_tar_probs_dict,class_label ='digit 9',fig_saving=1,save_path=save_img_path)

#index= getKeysByValue(translation_tar_probs_dict,np.float32(0.928))
###############################
######## rotation #############
# Set the target label as 8
target_number = 9
target_labels = np.zeros(original_labels.shape)
target_labels[:, target_number] = 1

target_labels

img_gradient = tf.gradients(cross_entropy, x)[0]
perturbation = 0.5 #The amount to wiggle towards the gradient of target class.
steps = 20

adversarial_img = original_images.copy()
for i in range(0, steps):
    gradient = img_gradient.eval({x: adversarial_img, y_: target_labels, keep_prob: 1.0})
    #Update using value of gradient
    adversarial_img = adversarial_img - perturbation * gradient
    prediction = tf.argmax(y,1)
    prediction_val = prediction.eval(feed_dict={x: adversarial_img, keep_prob: 1.0}, session=sess)
    print("\n step: %d" %(i+1))
    print("predictions", prediction_val)
    probabilities = y
    probabilities_val = probabilities.eval(feed_dict={x: adversarial_img, keep_prob: 1.0}, session=sess)
    print('Probabilities is digit {}: {}\n'.format(original_number,np.round(probabilities_val[:, original_number],3)))
    print('Probabilities is digit {}: {}\n'.format(target_number,np.round(probabilities_val[:, target_number],3)))


def rotation_f(input, d_angle=15, n_images=10):
    image = tf.Variable(tf.zeros((28, 28)))
    #rotated_adv_img = np.empty([0,2])
    rotated_adv_img = np.empty([0,28*28])
    for i in range(n_images):
        adv = input[i].reshape([28,28])
        degree_angle = d_angle  # In degrees
        radian = degree_angle * np.pi / 180
        angle = tf.placeholder(tf.float32, ())
        rotated_image = tf.contrib.image.rotate(image, angle)
        rotated_img = rotated_image.eval(feed_dict={image: adv, angle: radian})
        rotated_flatten_img = rotated_img.reshape((-1,28*28))
        rotated_adv_img = np.append(rotated_adv_img,rotated_flatten_img, axis=0)
    return rotated_adv_img


def multiple_rotation(input, start_angle=0, end_angle=2,n_images=10 ,correct_class=None):
    image = tf.Variable(tf.zeros((28, 28)))
    #rotated_adv_img = np.empty([0,2])
    #rotated_adv_img = np.empty([0,28*28])
    p = dict()
    for i in range(n_images):
        adv = input[i].reshape([28,28])
        for angle_index in range(start_angle,end_angle):
            degree_angle = angle_index  # In degrees
            radian = degree_angle * np.pi / 180
            angle = tf.placeholder(tf.float32, ())
            rotated_image = tf.contrib.image.rotate(image, angle)
            rotated_img = rotated_image.eval(feed_dict={image: adv, angle: radian})
            rotated_flatten_img = rotated_img.reshape((-1,28*28))
            probabilities = y
            probabilities_val = probabilities.eval(feed_dict={x: rotated_flatten_img, keep_prob: 1.0}, session=sess)
            probabilities_val = probabilities_val.tolist()
            #p.append(probabilities_val[correct_class])
            #rotated_adv_img = np.append(rotated_adv_img,rotated_flatten_img, axis=0)
            prob_value = probabilities_val[0][correct_class]
            p[(i, angle_index)] = prob_value
    return p


def plot_multi_rotation_prob(input, probabilities_dict, img_name='Image', correct_class = None, n_images=10, start_angle=0, end_angle=91, fig_saving=0, fig_size=[2093, 905], save_path=None):
    plt.close('all')
    for i in range(0, n_images):
        list_p = list()
        for j in range(start_angle,end_angle):
            list_p.append(probabilities_dict[(i,j)])
        img = input[i].reshape([28,28])
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.sca(ax1)
        ax1.imshow(img)     #ax1.imshow(img, cmap='gray')
        plt.title(img_name)

        plt.sca(ax2)
        # ax = fig.add_subplot(111)
        index_max, value_max = max(enumerate(list_p), key=operator.itemgetter(1))
        x = [i for i in range(end_angle-start_angle)]
        y = list_p
        line, = ax2.plot(x, y)

        ymax = value_max
        # xpos = y.index(ymax)
        xmax = index_max

        ax2.annotate("angle: " + str(index_max) + ",prob: " + str(round(value_max, 3)), xy=(xmax, ymax),
                     xytext=(xmax, ymax),
                     arrowprops=dict(facecolor='red', shrink=0.05), )

        ax2.set_ylim(0, 1)
        plt.xlabel('Angle of rotation')
        plt.ylabel('Classification probability for ' + str(correct_class))
        plt.show()
        if(fig_saving):
            fig1 = plt.gcf()
            DPI = fig1.get_dpi()
            width,height=fig_size
            fig1.set_size_inches(width / float(DPI), height / float(DPI))
            filename = 'figure'+str(i)+'-2.png'
            path = save_path
            fig1.savefig(path+filename, bbox_inches='tight');

    return 0

def plot_multi_rotation_prob_2(input,n_images=10, pro_dict_1=None, pro_dict_2=None, img_title='img', title_1="ori", title_2="adv", range_number = 91, fig_saving=0, fig_size=[2093, 905], save_path=None):
    plt.close('all')
    for i in range(0, n_images):
        list_p_1 = list()
        for j in range(range_number):
            list_p_1.append(pro_dict_1[(i,j)])

        list_p_2 = list()
        for j in range(range_number):
            list_p_2.append(pro_dict_2[(i,j)])
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.sca(ax1)
        img = input[i].reshape([28, 28])
        ax1.imshow(img)
        plt.title(img_title)
        plt.sca(ax2)
        y=list_p_1
        z=list_p_2
        #plt.figure()
        #plt.xlim(i for i in range(range_number+1))
        ax2.set_ylim(0, 1.1)
        ax2.plot(y,label=title_1,color = "green")
        ax2.plot(z,label=title_2,color = "red")
        plt.xlabel("angle of rotation")
        plt.ylabel("probabilities")
        plt.legend()
        plt.show()
        if fig_saving:
            fig1 = plt.gcf()
            DPI = fig1.get_dpi()
            wid, hei = fig_size
            fig1.set_size_inches(wid / float(DPI), hei / float(DPI))
            filename = 'fig_' + str(np.random.randint(low=1, high=1000, size=1)) + '-2.png'
            path = save_path
            fig1.savefig(path+filename, bbox_inches='tight')

'''
for i in range(0, 10):
    plt.figure(figsize=(2, 2))
    plt.axis('off')
    plt.imshow(rotated_adv_img[i].reshape([28, 28]))
    plt.show()

'''
def prediction(adv_input):
    # Predict the model on the rotated adversarial samples
    prediction = tf.argmax(y, 1)
    prediction_val = prediction.eval(feed_dict={x: adv_input, keep_prob: 1.0}, session=sess)
    print("predictions", prediction_val)
    probabilities = y
    probabilities_val = probabilities.eval(feed_dict={x: adv_input, keep_prob: 1.0}, session=sess)
    print("probabilities\n", np.round(probabilities_val, 3))


def plot_func(ori_input,adv_input, modified_input, correct_class=None, target_class=None,angle=90,wid=1075.0, hei=499.0, save_path=None):
    # Predict the model on the rotated adversarial samples
    prediction = tf.argmax(y, 1)
    prediction_val = prediction.eval(feed_dict={x: adv_input, keep_prob: 1.0}, session=sess)
    print("[adversarial] predictions", prediction_val)
    probabilities = y
    probabilities_ori = probabilities.eval(feed_dict={x: ori_input, keep_prob: 1.0}, session=sess)
    probabilities_val = probabilities.eval(feed_dict={x: adv_input, keep_prob: 1.0}, session=sess)
    probabilities_mod = probabilities.eval(feed_dict={x: modified_input, keep_prob: 1.0}, session=sess)
    print("[adversarial] probabilities\n", np.round(probabilities_val, 3))

    n,_ = adv_input.shape
    #n=1
    plt.close('all')
    for i in range(0,n):
        fig = plt.figure()
        ax1 = fig.add_subplot(321)
        ax1.imshow(ori_input[i].reshape([28,28]), cmap='gray')
        plt.title("Original Image")

        ax2 = plt.subplot(322)
        x_axis = range(10)
        y_axis = probabilities_ori[i]
        barlist = ax2.bar(x_axis,y_axis, align='center')
        barlist[target_class].set_color('r')
        plt.sca(ax2)
        plt.ylim([0, 1.1])
        plt.xticks(range(10))
        for rect in barlist:
            height = rect.get_height()
            ax2.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, '%0.3f' % float(height), ha='center', va='bottom')

        ax3 = fig.add_subplot(323)
        ax3.imshow(adv_input[i].reshape([28,28]), cmap='gray')
        plt.title("Adversarial Image")

        ax4 = plt.subplot(324)
        x_axis = range(10)
        y_axis = probabilities_val[i]
        barlist = ax4.bar(x_axis,y_axis, align='center')
        barlist[target_class].set_color('r')
        plt.sca(ax4)
        plt.ylim([0, 1.1])
        plt.xticks(range(10))
        for rect in barlist:
            height = rect.get_height()
            ax4.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, '%0.3f' % float(height), ha='center', va='bottom')


        ax5 = fig.add_subplot(325)
        ax5.imshow(modified_input[i].reshape([28,28]), cmap='gray')
        plt.title("Rotated Image, angle: "+str(angle))

        ax6 = plt.subplot(326)
        x_axis = range(10)
        y_axis = probabilities_mod[i]
        barlist = ax6.bar(x_axis,y_axis, align='center')
        barlist[target_class].set_color('r')
        plt.sca(ax6)
        plt.ylim([0, 1.1])
        plt.xticks(range(10))
        for rect in barlist:
            height = rect.get_height()
            ax6.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, '%0.3f' % float(height), ha='center', va='bottom')

        fig1 = plt.gcf()
        DPI = fig1.get_dpi()
        fig1.set_size_inches(wid / float(DPI), hei / float(DPI))
        filename = 'gen_figure'+str(i)+'-2.png'
        path = save_path
        fig1.savefig(path+filename, bbox_inches='tight');
        #plt.show()


def plot_imgages(img_1, img_2, title_1='adversarial image', title_2 ='rotated image'):
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(img_1)
    a.set_title(title_1)
    a = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(img_2)
    imgplot.set_clim(0.0, 0.7)
    a.set_title(title_2)

########################
#### Rotation only#####
### mutil-rotation angle
startTime = time.time()
prob_list = multiple_rotation(adversarial_img,start_angle=0,end_angle=30,n_images=number_images,correct_class=original_number)

#prob_list_2 = multiple_rotation(original_images,start_angle=0,end_angle=30,n_images=number_images,correct_class=original_number)
prob_list_3 = multiple_rotation(adversarial_img,start_angle=0,end_angle=30,n_images=number_images,correct_class=target_number)
elapsedTime = time.time() - startTime
print("elapsed Time: ", elapsedTime)

save_img_path = '../your_path/'

plot_multi_rotation_prob(input=adversarial_img, probabilities_dict=prob_list, correct_class=original_number, img_name='Adverarial Image, targeted: 4',
                         n_images = number_images, save_path=save_img_path, start_angle=0, end_angle=30,fig_saving=1)
#plot_multi_rotation_prob(input=original_images, probabilities_dict=prob_list_2, correct_class=original_number, img_name='Original Image',
#                         n_images = number_images,fig_size=[1075.0,499.0], save_path=save_img_path, start_angle=0, end_angle=30,fig_saving=1)

plot_multi_rotation_prob(input=adversarial_img, probabilities_dict=prob_list_3, correct_class=target_number, img_name='Adverarial Image, targeted: 4',
                         n_images = number_images,fig_size=[1075.0,499.0], save_path=save_img_path, start_angle=0, end_angle=30,fig_saving=1)

plot_multi_rotation_prob_2(input=adversarial_img,n_images=10, pro_dict_1=prob_list, pro_dict_2=prob_list_3, img_title='Adverarial Image, targeted: 4', title_1="digit 0",
                           title_2="digit 4", range_number = 30, fig_saving=1, save_path=save_img_path,fig_size=[1075.0,499.0])
### single rotation angle
plt.close('all')
rotation_angle = 10
rotated_adv_img = rotation_f(adversarial_img, d_angle=rotation_angle)
prediction(rotated_adv_img)
plot_func(ori_input = original_images,adv_input = adversarial_img, modified_input= rotated_adv_img,correct_class=original_number,target_class=target_number,angle=rotation_angle,wid=1267,hei=1442,save_path=save_img_path)

# Show the original images, correct, predicted label and confidence
for i in range(0, 10):
        print('Correct label', np.argmax(original_labels[i]))
        print('Predicted label:', prediction_val[i])
        print('Confidence:', np.max(probabilities_val[i]))
        plt.figure(figsize=(2, 2))
        plt.axis('off')
        plt.imshow(rotated_adv_img[i].reshape([28, 28]))
        plt.show()

################################
#### translation ###############

translated_images_list = translation_f(adversarial_img, input_tx=1, input_ty=1, n_images = 100)
#show_image_list(translated_images_list,number_images=10)
prediction = tf.argmax(y,1)
#prediction_val = sess.run(prediction,feed_dict={x: original_images, keep_prob: 1.0})
prediction_val = prediction.eval(feed_dict={x: translated_images_list, keep_prob: 1.0}, session=sess)
print("Predictions", prediction_val)
probabilities = y
probabilities_val = probabilities.eval(feed_dict={x: translated_images_list, keep_prob: 1.0}, session=sess)
#print ("probabilities\n", np.round(probabilities_val,3))
print('Probabilities is digit {}: {}\n'.format(original_number, np.round(probabilities_val[:, original_number], 3)))
print('Probabilities is digit {}: {}\n'.format(target_number, np.round(probabilities_val[:, target_number], 3)))


################################
##### translation & rotation####
translated_images_list = translation_f(adversarial_img, input_tx=3, input_ty=3, n_images = 10)
startTime = time.time()
combination_prob_list = multiple_rotation(translated_images_list,start_angle=0,end_angle=30,n_images=number_images,correct_class=original_number)

#plot_multi_rotation_prob(input=adversarial_img, probabilities_dict=combination_prob_list, correct_class=original_number,
#                         img_name='Adverarial Image, targeted: 4', n_images = number_images, save_path=save_img_path, start_angle=0, end_angle=30,fig_saving=1)
elapsedTime = time.time() - startTime
elapsedTime
startTime = time.time()
combination_prob_list_2 = multiple_rotation(translated_images_list,start_angle=0,end_angle=30,n_images=number_images,correct_class=target_number)
elapsedTime = time.time() - startTime
elapsedTime

save_img_path = '../your_path/'

#plot_multi_rotation_prob(input=adversarial_img, probabilities_dict=combination_prob_list_2, correct_class=target_number,
#                         img_name='Adverarial Image, targeted: 4', n_images = number_images, save_path=save_img_path, start_angle=0, end_angle=30,fig_saving=1)


plot_multi_rotation_prob_2(input=adversarial_img,n_images=10, pro_dict_1=combination_prob_list, pro_dict_2=combination_prob_list_2, img_title='Adverarial Image, targeted: 4', title_1="digit 0",
                           title_2="digit 4", range_number = 30, fig_saving=1, save_path=save_img_path,fig_size=[1075.0,499.0])

