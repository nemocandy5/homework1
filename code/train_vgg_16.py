import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
import sys
from tqdm import tqdm  # bar
from data_loader import load_data
sys.path.append(r"/home/nemocandy5/anaconda3/lib/python3.6/models-master/research/slim/preprocessing"); import vgg_preprocessing

# Load data
IMAGE = 'dataset/frames/'
LABEL = 'dataset/labels/'

train_image_paths, train_labels = load_data(image_folder=IMAGE, label_folder=LABEL, label_type='obj', left_right=['left', 'right'], with_head=True, ratio=0.2)
val_image_paths, val_labels = load_data(image_folder=IMAGE, label_folder=LABEL, label_type='obj', left_right=['left', 'right'], with_head=True, ratio=0.2)
test_image_paths, test_labels = load_data(image_folder=IMAGE, label_folder=LABEL, label_type='obj', left_right=['left', 'right'], with_head=True, ratio=0.2)


#Define the parameters
pretrained_model = 'pretrained_model/vgg_16.ckpt'
train_model = 'model/hand_obj_vgg_16_model.ckpt'
n_class = 24  # object categories (24 classes, including free)
batch_size = 32
epoch = 20
learning_rate = 0.00001
dropout_keep_prob = 0.5
weight_decay = 5e-4


graph = tf.Graph()
with graph.as_default():

    VGG = tf.contrib.slim.nets.vgg	
    training = tf.placeholder(dtype=tf.bool, name='training') # A variable to know training or not.
    image_path = tf.placeholder(dtype=tf.string, shape=(None,), name='image_path')
    labels = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')
    
    # Dataset API
    dataset = tf.contrib.data.Dataset.from_tensor_slices((image_path, labels)) 
    dataset = dataset.map(lambda image_path, label: data_labeling(image_path, label, training))
    dataset = dataset.shuffle(buffer_size=10000)
    batched_dataset = dataset.batch(batch_size)
    iterator = tf.contrib.data.Iterator.from_structure(batched_dataset.output_types, batched_dataset.output_shapes)
    #batch
    image_batch, label_batch = iterator.get_next()
    dataset_init_op = iterator.make_initializer(batched_dataset) 
    
    with slim.arg_scope(VGG.vgg_arg_scope(weight_decay=weight_decay)):
        logits, _ = VGG.vgg_16(image_batch, num_classes=n_class, is_training=training, dropout_keep_prob=dropout_keep_prob)
    
    variables = tf.contrib.framework.get_variables_to_restore(exclude=['vgg_16/fc8'])
    initial_function = tf.contrib.framework.assign_from_checkpoint_fn(pretrained_model, variables)

    fc8_variables = tf.contrib.framework.get_variables('vgg_16/fc8')
    fc8_initial = tf.variables_initializer(fc8_variables)
    
    tf.losses.sparse_softmax_cross_entropy(labels=label_batch, logits=logits)
    loss = tf.losses.get_total_loss()
    
    fc8_optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    fc8_train_op = fc8_optimizer.minimize(loss, var_list=fc8_variables)
    full_train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    prediction = tf.equal(tf.to_int32(tf.argmax(logits, 1)), label_batch)
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))    
    
    saver = tf.train.Saver()
	
def data_labeling(image_path, label, training):
    # Load image
    image_string = tf.read_file(image_path)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    # Preprocessing the image
    preprocessed_image = tf.cond(training, true_fn=lambda: vgg_preprocessing.preprocess_image(image, 224, 224, training=True), false_fn=lambda: vgg_preprocessing.preprocess_image(image, 224, 224, training=False))
    return preprocessed_image, label

# Training
def evaluate(sess, loss, prediction, dataset_init_op, feed_dict):
    
    sess.run(dataset_init_op, feed_dict=feed_dict)
    data_loss = 0
    n_correct = 0
    n_samples = 0
    
    while True:
        try:
            _loss, _correct_prediction = sess.run([loss, prediction], feed_dict={training: False})
			
			data_loss = data_loss + _loss
            n_correct += _correct_prediction.sum() 
            n_samples += _correct_prediction.shape[0] 
            
        except tf.errors.OutOfRangeError:
            break

    data_loss = data_loss / n_samples
    acc = n_correct / n_samples

    return data_loss, acc

sess = tf.Session(graph=graph)

initial_function(sess) # do this at training first time
sess.run(fc8_initial)  # do this at training first time

saver.save(sess, train_model)  # saver.restore(sess, train_model)

# Train 'fc8' layer
max_accuracy = 0.0
patience = 0

for epoch in tqdm(range(epoch)):
    # Run an epoch over the training data.
    print('-'*100)
    print('Starting epoch {}/{}'.format(epoch+1, epoch))

    sess.run(dataset_init_op, feed_dict={image_paths: train_image_paths, labels: train_labels, training: True})
    while True:
        try:
            _ = sess.run(fc8_train_op, feed_dict={training: True})
        except tf.errors.OutOfRangeError:
            break

    # performance
    train_loss, train_accuracy = evaluate(sess, loss, prediction, dataset_init_op, feed_dict={image_paths: train_image_paths, labels: train_labels, training: True})
    val_loss, val_accuracy = evaluate(sess, loss, prediction, dataset_init_op, feed_dict={image_paths: val_image_paths, labels: val_labels, training: False})
    
    print('[Train] loss: {} | accuracy: {}'.format(train_loss, train_accuracy))
    print('[Validation] loss: {} | accuracy: {}'.format(val_loss, val_accuracy))
    # Save checkpoint
    if val_accuracy > max_accuracy:
        patience = 0
        max_accuracy = val_accuracy
        save_path = saver.save(sess, train_model)
        print("Model updated and saved in file: %s" % save_path)
    else:
        patience += 1
        print('Model not improved at epoch {}/{}. Patience: {}/{}'.format(epoch+1, epoch, patience, max_patience))


# Train all layers
max_accuracy = 0.0
patience = 0
for epoch in tqdm(range(epoch)):
    # Run an epoch over the training data.
    print('-'*100)
    print('Starting epoch {}/{}'.format(epoch+1, epoch))

    sess.run(dataset_init_op, feed_dict={image_paths: train_image_paths, labels: train_labels, training: True})
    while True:
        try:
            _ = sess.run(full_train_op, feed_dict={training: True})    
        except tf.errors.OutOfRangeError:
            break

    # performance
    train_loss, train_accuracy = evaluate(sess, loss, prediction, dataset_init_op, feed_dict={image_paths: train_image_paths, labels: train_labels, training: True})  
    val_loss, val_accuracy = evaluate(sess, loss, prediction, dataset_init_op, feed_dict={image_paths: val_image_paths, labels: val_labels, training: False})
    
    print('[Train] loss: {} | accuracy: {}'.format(train_loss, train_accuracy))
    print('[Validation] loss: {} | accuracy: {}'.format(val_loss, val_accuracy))
    # Save checkpoint
    if val_accuracy > max_accuracy:
        patience = 0
        max_accuracy = val_accuracy
        save_path = saver.save(sess, train_model)
        print("Model updated and saved in file: %s" % save_path)
    else:
        patience += 1
        print('Model not improved at epoch {}/{}. Patience: {}/{}'.format(epoch+1, epoch, patience, max_patience))

# Testing
test_loss, test_accuracy = evaluate(sess, loss, prediction, dataset_init_op, feed_dict={image_paths: test_image_paths, labels: test_labels, training: False})

print('[Test] loss: {} | accuracy: {}'.format(test_loss, test_accuracy))
