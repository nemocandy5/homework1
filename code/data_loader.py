import os
import sys
import numpy as np

def load_data(image_folder='dataset/frames/', label_folder='dataset/labels', label_type='obj', left_right=['left', 'right'], with_head=False, ratio=0.2):
    
    train_head_image_paths, train_hand_image_paths, train_labels, val_head_image_paths, val_hand_image_paths, val_labels, test_head_image_paths, test_hand_image_paths, test_labels = [], [], np.array([])
    
    # training data
    for x in left_right:
        if with_head=True:
            _head_image_paths, _hand_image_paths, _labels = load_examples(image_folder=image_folder, label_folder=label_folder, data_type='train', label_type=label_type, hand_type=hand_type, with_head=True)
            _train_head_image_paths, _val_head_image_paths, _, _ = split(_head_image_paths, _labels, test_size=ratio)
                
            train_head_image_paths += _train_head_image_paths
            val_head_image_paths += _val_head_image_paths  
        else:
            _hand_image_paths, _labels = load_examples(image_folder=image_folder, label_folder=label_folder, data_type='train', label_type=label_type, hand_type=hand_type, with_head=False)
        
        _train_hand_image_paths, _val_hand_image_paths, _train_labels, _val_labels = split(_hand_image_paths, _labels, test_size=ratio)
        train_hand_image_paths += _train_hand_image_paths
        val_hand_image_paths += _val_hand_image_paths
        
        train_labels = np.concatenate([train_labels, _train_labels])
		train_labels = train_labels.astype(np.int32)
        val_labels = np.concatenate([val_labels, _val_labels])
		val_labels = val_labels.astype(np.int32)
    
    # testing data
    for x in left_right:
        if with_head=True:
            _test_head_image_paths, _test_hand_image_paths, _test_labels = load_examples(image_folder=image_folder, label_folder=label_folder, data_type='test', label_type=label_type, hand_type=hand_type, with_head=True)           
            test_head_image_paths += _test_head_image_paths
        else:
            _test_hand_image_paths, _test_labels = load_examples(image_folder=image_folder, label_folder=label_folder, data_type='test', label_type=label_type, hand_type=hand_type, with_head=False)
        
        test_hand_image_paths += _test_hand_image_paths
        test_labels = np.concatenate([test_labels, _test_labels])
		test_labels = test_labels.astype(np.int32)

    
    return train_head_image_paths, train_hand_image_paths, train_labels, \
           val_head_image_paths, val_hand_image_paths, val_labels, \
           test_head_image_paths, test_hand_image_paths, test_labels
		   
def load_image(image_folder='dataset/frames/', data_type='train', label_type='obj', image_type = 'left'):
    
    if image_type == 'left':
        image_type = 'Lhand'
    elif image_type == 'right':
        image_type = 'Rhand'
    else:
        image_type = 'head'
    
    image_paths = []
    
    # training and testing
    if data_type == 'train':
        index, _ = number(index=0)
    elif data_type == 'test':
        _, index = number(index=0)
        
        split_num = {'lab': 0, 'office': 0, 'house': 0}
        
        for x in index:
            scene_type = x[0]
            split_num[scene_type] += 1
            
        for i, x in enumerate(index):
            scene_type, split_id = x
            tmp_split_id = split_id % split_num[scene_type] 
            split_id = tmp_split_id if tmp_split_id > 0 else split_num[scene_type]
            index[i] = (scene_type, split_id) 

    for x in index:
        scene_type, split_id = x
        target_folder_path = os.path.join(image_folder, data_type, scene_type, str(split_id), image_type)
        file_names = os.listdir(target_folder_path)
        image_paths.extend([os.path.join(target_folder_path, file_name) for file_name in file_names])
        
    return image_paths
	
def number(index):
    train_index=[]
    test_index =[]
    if index == 0:
        for i in range(1,9):
            if i <= 4:
                train_index.append(('lab',i))
            else:
                test_index.append(('lab',i))
        for i in range(1,7):
            if i <= 3:
                train_index.append(('office',i))
            else:
                test_index.append(('office',i))
        for i in range(1,7):
            if i <= 3:
                train_index.append(('house',i))
            else:
                test_index.append(('house',i))
    elif index == 1:
        for i in range(1,9):
            train_index.append(('lab',i))
        for i in range(1,7):
            train_index.append(('office',i))
        for i in range(1,7):
            test_index.append(('house',i))
    else:
        raise ValueError('error setting index')

    return train_index, test_index


        
def load_labels(label_folder='dataset/labels', data_type='train', label_type='obj', hand_type = 'left'):
    
    labels = []
    
    # splitted the data to train and test.
    if data_type == 'train':
        index, _ = number(index=0)
    elif data_type == 'test':
        _, index = number(index=0)

    for x in index:
        scene_type, split_id = x
        label_npy_path = os.path.join(label_folder, scene_type, '{}_{}{}.npy'.format(label_type, hand_type, split_id))
        label_npy = np.load(label_npy_path)
        labels.append(label_npy)

    labels = np.concatenate(labels)
    
    return labels

def load_examples(image_folder='dataset/frames/', label_folder='dataset/labels', data_type='train', label_type='obj', hand_type='left', with_head=False):
    
    hand_image_paths = load_image(image_folder=image_folder, data_type=data_type, label_type=label_type, image_type=hand_type)
    
    if with_head:
        head_image_paths = load_image(image_folder=image_folder, data_type=data_type, label_type=label_type, image_type='head')
    
    labels = load_labels(label_folder=label_folder, data_type=data_type, label_type=label_type, hand_type=hand_type)
    labels = labels.astype(np.int32)
    
    if with_head:
        return head_image_paths, hand_image_paths, labels
    else:
        return hand_image_paths, labels
    
def split(image_paths, labels, test_size=0.2):
    num_lab = 0
    num_office = 0
    num_house = 0
        
    for image_path in image_paths:
        
        f_names = image_path.split('/')
        frame_folder_idx = [i for i, name in enumerate(f_names) if name == 'frames'][0]
        scene = f_names[frame_folder_idx+2]
        
        if scene == 'lab': num_lab += 1
        elif scene == 'office': num_office += 1
        elif scene == 'house': num_house += 1
            
    lab_image_paths = image_paths[:num_lab]
    lab_labels = labels[:num_lab]
    
    office_image_paths = image_paths[num_lab:num_lab+num_office]
    office_labels = labels[num_lab:num_lab+num_office]
    
    house_image_paths = image_paths[num_lab+num_office:num_lab+num_office+num_house]
    house_labels = labels[num_lab+num_office:num_lab+num_office+num_house] 
    
    lab_train_size = round((1 - test_size)*num_lab)
    office_train_size = round((1 - test_size)*num_office)
    house_train_size = round((1 - test_size)*num_house)
    
    train_image_paths = lab_image_paths[:lab_train_size] + office_image_paths[:office_train_size] + house_image_paths[:house_train_size]
    train_labels = np.concatenate([lab_labels[:lab_train_size], office_labels[:office_train_size], house_labels[:house_train_size]])
    
    test_image_paths = lab_image_paths[lab_train_size:] + office_image_paths[office_train_size:] + house_image_paths[house_train_size:]
    test_labels = np.concatenate([lab_labels[lab_train_size:], office_labels[office_train_size:], house_labels[house_train_size:]])
    
    return train_image_paths, test_image_paths, train_labels, test_labels
