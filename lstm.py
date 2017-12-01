from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import pickle
import argparse

import tensorflow as tf
import numpy as np

from base_model import BaseModel, CaptionGenerator
from prepare_data import DataSet

def main(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--phase', help='Phase: Can be train, val or test')
    parser.add_argument('--load', action='store_true', default=False, help='Turn on to load the pretrained model')
    
    parser.add_argument('--cnn_model', default='resnet152', help='CNN model to use: Can be vgg16, resnet50, resnet101 or resnet152')
    parser.add_argument('--cnn_model_file', help='Tensorflow model file for the chosen CNN model')
    parser.add_argument('--load_cnn_model', action='store_true', default=True, help='Turn on to load the pretrained CNN model')
    parser.add_argument('--train_cnn', default=False, help='Turn on to jointly train CNN and RNN. Otherwise, only RNN is trained')

    parser.add_argument('--images_dir', default='./train/images/', help='Directory containing the COCO train2014 images')
    parser.add_argument('--caption_file', default='./train/captions_train2014.json', help='JSON file storing the captions for COCO train2014 images')
    parser.add_argument('--test_result_file', default='./results.json', help='File to store the testing results')

    parser.add_argument('--word_table_file', default='./word_table.pickle', help='Temporary file to store the word table')
    parser.add_argument('--max_sent_len', type=int, default=30, help='Maximum length of the generated caption')

    parser.add_argument('--save_dir', default='./models/', help='Directory to contain the trained model')
    parser.add_argument('--save_period', type=int, default=2000, help='Period to save the trained model')
    
    parser.add_argument('--solver', default='adam', help='Optimizer to use: Can be adam, momentum, rmsprop or sgd') 
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (for some optimizers)') 
    parser.add_argument('--decay', type=float, default=0.9, help='Decay (for some optimizers)') 
    parser.add_argument('--batch_norm', action='store_true', default=False, help='Turn on to use batch normalization')  

    parser.add_argument('--num_lstm', type=int, default=1, help='Number of LSTMs to use: Can be 1 or 2')
    parser.add_argument('--dim_hidden', type=int, default=1000, help='Dimension of the hidden state in each LSTM')
    parser.add_argument('--dim_embed', type=int, default=300, help='Dimension of the word embedding')
    parser.add_argument('--dim_dec', type=int, default=1000, help='Dimension of the vector used for word generation')
    parser.add_argument('--num_init_layers', type=int, default=2, help='Number of layers in the MLP for initializing the LSTMs')
    parser.add_argument('--init_lstm_with_fc_feats', action='store_true', default=False, help='Turn on to initialize the LSTMs with fc7 feats of VGG16 net. Only useful if VGG16 is used')

    args = parser.parse_args()

    with tf.Session() as sess:    
        # training phase  
        if args.phase == 'train':

            train_data = DataSet(images_dir=args.images_dir, 
                                caption_file=args.caption_file, 
                                max_sent_len=args.max_sent_len, 
                                batch_size=args.batch_size,
                                save_file=args.word_table_file)
            train_data.build_word_table(args.dim_embed)
            train_data.reset()

            model = CaptionGenerator(args, 'train')
            model.build(train_data.num_words, 
                        train_data.word2vec, 
                        train_data.idx2word)

            sess.run(tf.global_variables_initializer())
    
            if args.load:
                model.load(sess)
            elif args.load_cnn_model:
                model.load2(args.cnn_model_file, sess)

            model.train(sess, train_data)
        else: 
            test_data = DataSet(images_dir=args.images_dir,
                                save_file=args.word_table_file)
            test_data.load()

            model = CaptionGenerator(args, 'test')

            # model.build(test_data.num_words, 
            #             test_data.word2vec, 
            #             test_data.idx2word)

            model.load(sess)

            model.test(sess, test_data)

if __name__=="__main__":
     main(sys.argv)

