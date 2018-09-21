import keras.backend as K
import keras
from keras.models import load_model
import tensorflow as tf
import time
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans import utils

#from cleverhans.attacks import CarliniWagnerL2

import csv
import scipy.io
import glob
import numpy as np
import sys



# parameters
dataDir = './training_raw/'
FS = 300
WINDOW_SIZE = 30*FS     # padding window for CNN
classes = ['A', 'N', 'O','~']

keras.layers.core.K.set_learning_phase(0)
# loading model
sess = tf.Session()
K.set_session(sess)

print("Loading model")    
model = load_model('./ResNet_30s_34lay_16conv.hdf5')
#model = load_model('weights-best_k0_r0.hdf5')

wrap = KerasModelWrapper(model)

x = tf.placeholder(tf.float32, shape=(None, 9000, 1))
y = tf.placeholder(tf.float32, shape=(None, 4))

# load groundTruth
print("Loading ground truth file")   
csvfile = list(csv.reader(open('REFERENCE-v3.csv')))
files = sorted(glob.glob(dataDir+"*.mat"))

def preprocess(x, maxlen):
    x =  np.nan_to_num(x)
#    x =  x[0, 0:min(maxlen,len(x))]
    x =  x[0, 0:maxlen]
    x = x - np.mean(x)
    x = x / np.std(x)
    
    tmp = np.zeros((1, maxlen))
#    print(x.shape)
    tmp[0, :len(x)] = x.T  # padding sequence
    x = tmp
#    print(x.shape)
    x = np.expand_dims(x, axis=2)  # required by Keras
#    print(x.shape)
    del tmp
    
    return x
def op_concate(x):
    data_len = 9000
    p = np.random.randint(data_len)
    x1 = [x[0, 0:p]]
    x2 = [x[0, p:]]
    return np.append(x2, x1, axis=1)


preds = model(x)

## Loading time serie signals
id = int(sys.argv[1])
count = id-1
record = "A{:05d}".format(id)
local_filename = dataDir+record
# Loading
mat_data = scipy.io.loadmat(local_filename)
print('Loading record {}'.format(record))    
#    data = mat_data['val'].squeeze()
data = mat_data['val']
data = preprocess(data, WINDOW_SIZE)

ground_truth_label = csvfile[count][1]
ground_truth = classes.index(ground_truth_label)
print('Ground truth:{}'.format(ground_truth))

X_test=np.float32(data)
new_X_test = np.repeat(data, 5, axis=0)


Y_test = np.zeros((1, 1))
Y_test[0,0] = ground_truth
new_Y_test = np.zeros((5, 1))
new_Y_test = np.repeat(ground_truth, 5, axis=0)
Y_test = utils.to_categorical(Y_test, num_classes=4)


target_a = np.zeros((1, 1))
target_a = np.array([int(sys.argv[2])])
target_a = utils.to_categorical(target_a, num_classes=4)

start_time = time.time()
from EOT_g import EOT_L2
eotl2 = EOT_L2(wrap, sess=sess)
eotl2_params = {'y_target': target_a, 'learning_rate': 0.1, 'max_iterations': 1000}

adv_x = eotl2.generate(x, **eotl2_params)
adv_x = tf.stop_gradient(adv_x) # Consider the attack to be constant
#preds_adv = model(adv_x)
feed_dict = {x: X_test}
#adv_sample = sess.run(adv_x, feed_dict=feed_dict)
adv_sample = adv_x.eval(feed_dict=feed_dict, session = sess)

print("time used:", time.time()-start_time)
perturb = adv_sample-X_test

correct = 0
attack_success = 0

for _ in range(100):
    #new_X_test = op_concate(X_test)
    #prob_ori = model.predict(new_X_test)
    prob_att = model.predict(op_concate(perturb)+X_test)
    #if np.argmax(prob_ori) == ground_truth:
        #correct = correct + 1
    if np.argmax(prob_att) != ground_truth:
        attack_success = attack_success + 1
#print("correct:", correct)
print("attack success times:", attack_success)

perturb_squeeze = np.squeeze(perturb, axis=2)
np.savetxt('./output/EOT_t30_f1_l2_A5T2.out', perturb_squeeze,delimiter=",")
prob = model.predict(adv_sample)
ann = np.argmax(prob)
ann_label = classes[ann]
print(ann)

'''
import matplotlib.pyplot as plt
plt.figure()
plt.plot(X_test[0,:,0])
plt.show()

plt.figure()
plt.plot(perturb[0,:,0]+X_test[0,:,0])
plt.show()
'''


#
#ymax = np.max(adv_sample)+0.5
#ymin = np.min(adv_sample)-0.5
#
#fig, axs = plt.subplots(1, 3, figsize=(20,5))
#
#axs[0].plot(X_test[0,:])
#axs[0].set_title('Original signal {}'.format(ground_truth_label))
#axs[0].set_ylim([ymin, ymax])
#axs[0].set_xlabel('index')
#axs[0].set_ylabel('signal value')
#
#axs[1].plot(adv_sample[0,:])
#axs[1].set_title('Adversarial signal {}'.format(ann_label))
#axs[1].set_ylim([ymin, ymax])
#axs[1].set_xlabel('index')
#axs[1].set_ylabel('signal value')
#
#axs[2].plot(adv_sample[0,:]-X_test[0,:])
#axs[2].set_title('perturbations')
#axs[2].set_ylim([ymin, ymax])
#axs[2].set_xlabel('index')
#axs[2].set_ylabel('signal value')
