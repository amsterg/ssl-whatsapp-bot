import tensorflow as tf
import numpy as np
import random
import time
import config
import data

class ChappieModel(object):
    """docstring for ChappieModel."""
    def __init__(self, forward_only, batch_size):
        self.forward_only = forward_only
        self.batch_size = batch_size

    def inference(self):
        print("Inference")
        w = tf.get_variable('proj_w', [config.hidden_size, config.vocab_dec])
        b = tf.get_variable('proj_b', [config.vocab_dec])
        self.output_projection = (w, b)

        def sampled_loss(labels,logits):
            labels = tf.reshape(labels,[-1,1])
            return tf.nn.sampled_softmax_loss(tf.transpose(w),b,labels,logits,config.num_samples,config.vocab_dec)


        self.softmax_loss_function = sampled_loss

        single_cell = tf.contrib.rnn.GRUCell(config.hidden_size)
        self.cell = tf.contrib.rnn.MultiRNNCell([single_cell for _ in range(config.num_layers)])

    def create_placeholders(self):
        print("Creating placeholders...")
        self.enc_inputs = []
        self.dec_inputs = []
        self.dec_masks = []
        for i in xrange(config.buckets[-1][0]):
            self.enc_inputs.append(tf.placeholder(dtype = tf.int32,shape = [None],name = 'encoder{}'.format(i)))
        for i in xrange(config.buckets[-1][1]+1):
            self.dec_inputs.append(tf.placeholder(dtype = tf.int32,shape = [None],name = 'decoder{}'.format(i)))
            self.dec_masks.append(tf.placeholder(dtype = tf.float32,shape = [None],name = 'masks{}'.format(i)))

        self.targets = [self.dec_inputs[i + 1]
                        for i in xrange(len(self.dec_inputs) - 1)]


    def lossNlayers(self):
        print("lossNlayers")
        def seq2seq_f(encoder_inputs,decoder_inputs,do_decode):
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                            encoder_inputs,decoder_inputs,self.cell,num_encoder_symbols=config.vocab_enc,
                            num_decoder_symbols=config.vocab_dec,
                            embedding_size=config.hidden_size,
                            output_projection=self.output_projection,
                            feed_previous=do_decode
                            )

        if self.forward_only:
            self.outputs,self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
            self.enc_inputs,self.dec_inputs,self.targets,self.dec_masks,config.buckets,
            lambda x, y: seq2seq_f(x, y, True),softmax_loss_function=self.softmax_loss_function
            )

            if self.output_projection:
                for b in xrange(len(config.buckets)):
                    self.outputs[b] = [tf.matmul(output, self.output_projection[0])+self.output_projection[1]
                                            for output in self.outputs[b]]
        else:
            self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
                                        self.enc_inputs,
                                        self.dec_inputs,
                                        self.targets,
                                        self.dec_masks,
                                        config.buckets,
                                        lambda x, y: seq2seq_f(x, y, False),
                                        softmax_loss_function=self.softmax_loss_function)


    def optimizer(self):
        print ("optimizer")
        with tf.variable_scope('training') as scope:
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            self.lr = config.lr

            if not self.forward_only:
                self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
                trainables = tf.trainable_variables()
                self.gradient_norms = []
                self.updated = []
                for bucket in xrange(len(config.buckets)):

                    clipped_grads, norm = tf.clip_by_global_norm(tf.gradients(self.losses[bucket],
                                                                 trainables),
                                                                 config.max_grad_norm)
                    self.gradient_norms.append(norm)
                    self.updated.append(self.optimizer.apply_gradients(zip(clipped_grads, trainables),
                                                            global_step=self.global_step))
        print("Done creating optimizer")



    def build_graph(self):
        self.create_placeholders()
        self.inference()
        self.lossNlayers()
        self.optimizer()
