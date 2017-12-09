
import os
import json
import pickle
import glob

import cv2
import numpy as np
import jieba

class DataSet(object):

    def __init__(self, images_dir, caption_file=None, max_sent_len=30, 
                batch_size=32, save_file='word_table.pickle', cut=False):

        self.images_dir = images_dir
        self.is_train = True if caption_file is not None else False

        self.save_file = save_file

        self.max_sent_len = max_sent_len
        self.batch_size = batch_size if self.is_train else 1

        self.cut = cut

        self.scale_shape = np.array([224, 224], np.int32)

        instance_id = 0
        if self.is_train:
            caption_annotations = json.load(open(caption_file, 'r'))
            
            self.idx_imagefile_caption = {}

            for caption_anna in caption_annotations:
                captions = caption_anna['caption']
                image_file_name = caption_anna['image_id']

                for caption in captions:
                    self.idx_imagefile_caption[instance_id] = {'image_file_name': image_file_name, 
                                                                'caption': self.process_caption(caption)}
                    instance_id += 1
        else:
            self.idx_imagefile = {}

            image_files = glob.glob(images_dir + '/*')

            for image_file in image_files:
                self.idx_imagefile[instance_id] = image_file.split(os.sep)[-1]
                instance_id += 1

            self.num_images = len(image_files)

        self.current_index = 0
        self.ids = list(range(instance_id))

        self.num_batches = int(len(self.ids) / self.batch_size)

    def next_batch(self):
        if self.current_index + self.batch_size > len(self.ids):
            self.reset()

        batch_ids = self.ids[self.current_index: self.current_index + self.batch_size]
        self.current_index += self.batch_size

        images = []
        image_file_names = []

        for idx in batch_ids:
            if self.is_train:
                image_file_name = self.idx_imagefile_caption[idx]['image_file_name']
            else:
                image_file_name = self.idx_imagefile[idx]

            image_file_names.append(image_file_name)

            # Load the image
            img = cv2.imread(self.images_dir + '/' + image_file_name)

            temp = img.swapaxes(0, 2)
            temp = temp[::-1]
            img = temp.swapaxes(0, 2)

            img = cv2.resize(img, (self.scale_shape[0], self.scale_shape[1]))

            img = img.astype(np.float32) - np.array([121.29502776, 113.9715251 , 106.10755275])

            images.append(img)

        images = np.array(images, dtype=np.float32)

        if not self.is_train:
            return image_file_names, images

        caption_vectors = []
        masks = []

        for idx in batch_ids:
            instance = self.idx_imagefile_caption[idx]

            # Convert the caption into an array of indices in word table
            caption = instance['caption']

            word_indices = np.zeros(self.max_sent_len).astype(np.int32)
            mask = np.zeros(self.max_sent_len)

            if self.cut:
                caption = caption.split()
                
            words = np.array([self.word2idx[w] for w in caption])

            word_indices[:len(words)] = words
            mask[:len(words)] = 1.0

            caption_vectors.append(word_indices)
            masks.append(mask)

        caption_vectors = np.array(caption_vectors)
        masks = np.array(masks)

        return image_file_names, images, caption_vectors, masks


    def reset(self):
        self.current_index = 0
        np.random.shuffle(self.ids)


    def process_caption(self, caption):
        truncated_caption = caption[:self.max_sent_len]

        # Segment the caption using Jieba
        if self.cut:
            seg_list = jieba.cut(truncated_caption, cut_all=False)
            return ' '.join(seg_list)

        return truncated_caption

    def load(self):
        self.idx2word, self.word2idx, self.word2vec, self.num_words = pickle.load(open(self.save_file, 'rb'))


    def build_word_table(self, dim_embed):
        if os.path.exists(self.save_file):
            self.idx2word, self.word2idx, self.word2vec, self.num_words = pickle.load(open(self.save_file, 'rb'))
            return

        self.idx2word = []
        self.word2idx = {}
        self.word2vec = {}
        self.word_freq = []
        self.dim_embed = dim_embed

        captions = [d['caption'] for d in self.idx_imagefile_caption.values()]

        for caption in captions:

            if self.cut:
                cpation = caption.split()

            for word in caption:
                if word not in self.word2vec:
                    # Why times 0.01 here??
                    self.word2vec[word] = 0.01 * np.random.randn(self.dim_embed)

        self.num_words = len(self.word2vec.keys())
        self.idx2word = sorted(self.word2vec.keys())

        for idx, word in enumerate(self.idx2word):
            self.word2idx[word] = idx

        if not os.path.exists(self.save_file):
            pickle.dump([self.idx2word, self.word2idx, self.word2vec, self.num_words], 
                open(self.save_file, 'wb'))

    def indices_to_sent(self, indices):
        """ Translate a vector of indicies into a sentence. """
        words = [self.idx2word[i] for i in indices]
        return ''.join(words)





