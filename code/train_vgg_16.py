# ### Load training, validation and testing data

from data_loader import load_dataset

IMAGE_FOLDER_PATH = 'dataset/frames/'
LABEL_FOLDER_PATH = 'dataset/labels/'

train_head_image_paths, train_hand_image_paths, train_labels, val_head_image_paths, val_hand_image_paths, val_labels, test_head_image_paths, test_hand_image_paths, test_labels = load_dataset(image_folder_path=IMAGE_FOLDER_PATH,
                                                                         label_folder_path=LABEL_FOLDER_PATH,
                                                                         label_type='obj',
                                                                         hand_types=['left', 'right'],
                                                                         with_head=True,
                                                                         validation_split_ratio=0.15)

# Only take hand image paths for baseline
train_image_paths =  train_hand_image_paths
val_image_paths = val_hand_image_paths
test_image_paths = test_hand_image_paths

#Use Tensorflow to build computational graph
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
import sys
sys.path.append(r"/home/nemocandy5/anaconda3/lib/python3.6/models-master/research/slim/preprocessing"); import vgg_preprocessing

#Define the pre-trained model
PRETRAINED_VGG_MODEL_PATH = 'pretrained_model/vgg_16.ckpt'
MODEL_PATH = 'model/hand_obj_vgg_16_model.ckpt'

num_classes = 24  # object categories (24 classes, including free)
batch_size = 32
max_epochs1 = 20
max_epochs2 = 20
max_patience = 3 # For early stopping
learning_rate1 = 0.0001   # 1e-3
learning_rate2 = 0.00001  # 1e-5
dropout_keep_prob = 0.5
weight_decay = 5e-4

def dataset_map_fn(image_path, label, is_training):
    # Load image
    image_string = tf.read_file(image_path)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    # Preprocess image
    preprocessed_image = tf.cond(is_training,
                                 true_fn=lambda: vgg_preprocessing.preprocess_image(image, 224, 224, is_training=True),
                                 false_fn=lambda: vgg_preprocessing.preprocess_image(image, 224, 224, is_training=False))
    return preprocessed_image, label

graph = tf.Graph()
with graph.as_default():
    # ---------------------------------------------------------------------
    # Indicates whether we are in training or in test mode
    # Since VGG16 has applied `dropout`, we need to disable it when testing.
    is_training = tf.placeholder(dtype=tf.bool, name='is_training')
    
    # Training, validation, testing data to feed in.
    image_paths = tf.placeholder(dtype=tf.string, shape=(None,), name='image_paths')
    labels = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')
    
    # Use " Dataset API " to automatically generate batch data by iterator.
    dataset = tf.contrib.data.Dataset.from_tensor_slices((image_paths, labels)) # Create a Dataset whose elements are slices.
    dataset = dataset.map(lambda image_path, label: dataset_map_fn(image_path, label, is_training))
    dataset = dataset.shuffle(buffer_size=10000)
    batched_dataset = dataset.batch(batch_size)
    
    # Now we define an iterator that can operator on dataset.
    # The iterator can be reinitialized by calling:
    # sess.run(dataset_init_op, feed_dict={image_paths=train_image_paths, labels=train_labels}) 
    # for 1 epoch on the training set.
    
    # Once this is done, we don't need to feed any value for images and labels
    # as they are automatically pulled out from the iterator queues.

    # A reinitializable iterator is defined by its structure. We could use the
    # `output_types` and `output_shapes` properties of dataset.
    # The dataset will be fed with training, validation or testing data.
	# Iterator.from_structure()  Creates a new, uninitialized Iterator with the given structure.
    iterator = tf.contrib.data.Iterator.from_structure(batched_dataset.output_types,
                                                       batched_dataset.output_shapes)
    
    # A batch of data to feed into the networks.
    batch_images, batch_labels = iterator.get_next()  # Returns a nexted structure of tf.Tensor containing the next element.
    dataset_init_op = iterator.make_initializer(batched_dataset) # Returns a tf.Operation that initializes this iterator on dataset.
    
    # ---------------------------------------------------------------------
    # Now that we have set up the data, it's time to set up the model.
    # For this example, we'll use VGG-16 pre-trained on ImageNet. We will remove the
    # last fully connected layer (fc8) and replace it with our own, with an
    # output size `num_classes`
    # We will first train the last layer for a few epochs.
    # Then we will train the entire model on our dataset for a few epochs.

    # Get the pre-trained model, specifying the num_classes argument to create a new
    # fully connected replacing the last one, called "vgg_16/fc8"
    # Each model has a different architecture, so "vgg_16/fc8" will change in another model.
    # Here, logits gives us directly the predicted scores we wanted from the images.
    # We pass a scope to initialize "vgg_16/fc8" weights with he_initializer
    vgg = tf.contrib.slim.nets.vgg
    with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
        logits, _ = vgg.vgg_16(batch_images, num_classes=num_classes, is_training=is_training,
                               dropout_keep_prob=dropout_keep_prob)
    
    # Restore only the layers up to fc7 (included)
    # Calling function `init_fn(sess)` will load all the pre-trained weights.
    variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['vgg_16/fc8'])
    init_fn = tf.contrib.framework.assign_from_checkpoint_fn(PRETRAINED_VGG_MODEL_PATH, variables_to_restore)

    # Initialization operation from scratch for the new "fc8" layers
    # `get_variables` will only return the variables whose name starts with the given pattern
    fc8_variables = tf.contrib.framework.get_variables('vgg_16/fc8')
    fc8_init = tf.variables_initializer(fc8_variables)
    
    # ---------------------------------------------------------------------
    # Using tf.losses, any loss is added to the tf.GraphKeys.LOSSES collection
    # We can then call the total loss easily
    tf.losses.sparse_softmax_cross_entropy(labels=batch_labels, logits=logits)
    loss = tf.losses.get_total_loss()
    
    # First we want to train only the reinitialized last layer fc8 for a few epochs.
    # We run minimize the loss only with respect to the fc8 variables (weight and bias).
    fc8_optimizer = tf.train.GradientDescentOptimizer(learning_rate1)
    fc8_train_op = fc8_optimizer.minimize(loss, var_list=fc8_variables)
    
    # Then we want to fine-tune the entire model for a few epochs.
    # We run minimize the loss only with respect to all the variables.
    full_optimizer = tf.train.GradientDescentOptimizer(learning_rate2)
    full_train_op = full_optimizer.minimize(loss)

    # Evaluation metrics
    prediction = tf.to_int32(tf.argmax(logits, 1))
    correct_prediction = tf.equal(prediction, batch_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))    
    
    # 'Saver' op to save and restore all the variables
    saver = tf.train.Saver()


# Start training
from tqdm import tqdm

def evaluate(sess, loss, correct_prediction, dataset_init_op, feed_dict):
    """
        Evaluation in training loop.
        Check the performance of the model on either train, val or test (depending on `dataset_init_op`)
        Note: The arguments are tensorflow operators defined in the graph.
    """
    
    # Initialize the correct dataset.
    sess.run(dataset_init_op, feed_dict=feed_dict)

    data_loss = 0
    num_correct = 0
    num_samples = 0
    
    # Evaluate on every batch.
    while True:
        try:
            # Disable `is_training` since we have `dropout` in VGG net.
            _loss, _correct_prediction = sess.run([loss, correct_prediction], feed_dict={is_training: False})

            data_loss += _loss
            num_correct += _correct_prediction.sum() # e.g: [True, False, True].sum() = 2
            num_samples += _correct_prediction.shape[0] # Batch size
            
        except tf.errors.OutOfRangeError:
            break

    data_loss = data_loss / num_samples
    acc = num_correct / num_samples

    return data_loss, acc

# --------------------------------------------------------------------------
# Now that we have built the graph and finalized it, we define the session.
# The session is the interface to *run* the computational graph.
# We can call our training operations with `sess.run(train_op)` for instance
sess = tf.Session(graph=graph)

#init_fn(sess) # load the pre-trained weights
#sess.run(fc8_init)  # initialize the new fc8 layer

saver.restore(sess, MODEL_PATH)  # saver.save(sess, MODEL_PATH)

# Train 'fc8' layer

max_acc = 0.0
patience = 0

# Update only the last layer for a few epochs.
for epoch in tqdm(range(max_epochs1)):
    # Run an epoch over the training data.
    print('-'*110)
    print('Starting epoch {}/{}'.format(epoch+1, max_epochs1))
    # Here we initialize the iterator with the training set.
    # This means that we can go through an entire epoch until the iterator becomes empty.
    sess.run(dataset_init_op, feed_dict={image_paths: train_image_paths,
                                         labels: train_labels,
                                         is_training: True})
    while True:
        try:
            _ = sess.run(fc8_train_op, feed_dict={is_training: True})
        except tf.errors.OutOfRangeError:
            break

    # Check performance every epoch
    train_loss, train_acc = evaluate(sess, loss, correct_prediction, dataset_init_op,
                                     feed_dict={image_paths: train_image_paths,
                                                labels: train_labels,
                                                is_training: True})
    
    val_loss, val_acc = evaluate(sess, loss, correct_prediction, dataset_init_op,
                                 feed_dict={image_paths: val_image_paths,
                                            labels: val_labels,
                                            is_training: False})
    
    print('[Train] loss: {} | accuracy: {}'.format(train_loss, train_acc))
    print('[Validation] loss: {} | accuracy: {}'.format(val_loss, val_acc))
    # Save checkpoint
    if val_acc > max_acc:
        patience = 0
        max_acc = val_acc
        save_path = saver.save(sess, MODEL_PATH)
        print("Model updated and saved in file: %s" % save_path)
    else:
        patience += 1
        print('Model not improved at epoch {}/{}. Patience: {}/{}'.format(epoch+1, max_epochs1, patience, max_patience))
    # Early stopping.
    """
    if patience > max_patience:
        print('Max patience exceeded. Early stopping.')
        break 
    """

# Train all layers

# Train the entire model for a few more epochs, continuing with the *same* weights.
max_acc = 0.0
patience = 0
for epoch in tqdm(range(max_epochs2)):
    # Run an epoch over the training data.
    print('-'*110)
    print('Starting epoch {}/{}'.format(epoch+1, max_epochs2))
    # Here we initialize the iterator with the training set.
    # This means that we can go through an entire epoch until the iterator becomes empty.
    sess.run(dataset_init_op, feed_dict={image_paths: train_image_paths,
                                         labels: train_labels,
                                         is_training: True})
    while True:
        try:
            _ = sess.run(full_train_op, feed_dict={is_training: True})    
        except tf.errors.OutOfRangeError:
            break

    # Check performance every epoch
    train_loss, train_acc = evaluate(sess, loss, correct_prediction, dataset_init_op,
                                     feed_dict={image_paths: train_image_paths,
                                                labels: train_labels,
                                                is_training: True})
    
    val_loss, val_acc = evaluate(sess, loss, correct_prediction, dataset_init_op,
                                 feed_dict={image_paths: val_image_paths,
                                            labels: val_labels,
                                            is_training: False})
    
    print('[Train] loss: {} | accuracy: {}'.format(train_loss, train_acc))
    print('[Validation] loss: {} | accuracy: {}'.format(val_loss, val_acc))
    # Save checkpoint
    if val_acc > max_acc:
        patience = 0
        max_acc = val_acc
        save_path = saver.save(sess, MODEL_PATH)
        print("Model updated and saved in file: %s" % save_path)
    else:
        patience += 1
        print('Model not improved at epoch {}/{}. Patience: {}/{}'.format(epoch+1, max_epochs1, patience, max_patience))
    # Early stopping.
    """
    if patience > max_patience:
        print('Max patience exceeded. Early stopping.')
        break
    """

# Testing

test_loss, test_acc = evaluate(sess, loss, correct_prediction, dataset_init_op,
                                 feed_dict={image_paths: test_image_paths,
                                            labels: test_labels,
                                            is_training: False})

print('[Test] loss: {} | accuracy: {}'.format(test_loss, test_acc))

