import os
import random
import numpy as np
import tensorflow as tf
from skel import ChappieModel
import data
import config
import sys
import time
def get_buckets():
    buckets = data.bucket_data()
    bucket_sizes = [len(buckets[b]) for b in xrange(len(config.buckets))]
    total_bucket_sizes = sum(bucket_sizes)
    buckets_scale = [sum(bucket_sizes[:i + 1]) / total_bucket_sizes
                           for i in xrange(len(bucket_sizes))]
    print("Bucket scale:\n", buckets_scale)
    return buckets, buckets_scale

def get_bucket_id(buckets_scale):
    rand_val = random.random()
    return min([i for i  in xrange(len(buckets_scale))
                        if buckets_scale[i] > rand_val])

def restore_saved_parameters(sess, saver):

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.cpt_path + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        print("Loading parameters for the Chatbot")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Initializing fresh parameters for the Chatbot")

def step(session,model, enc_inputs, dec_inputs, dec_masks,bucket_id, forward_only):
    input_feed = {}
    encoder_size,decoder_size = config.buckets[bucket_id]
    if len(enc_inputs) != encoder_size:
        raise ValueError("Encoder length must be equal to the one in bucket,"
                        " %d != %d." % (len(enc_inputs), encoder_size))
    if len(dec_inputs) != decoder_size:
        raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(dec_inputs), decoder_size))
    if len(dec_masks) != decoder_size:
        raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(dec_masks), decoder_size))
    for l in xrange(encoder_size):
      input_feed[model.enc_inputs[l].name] = enc_inputs[l]
    for l in xrange(decoder_size):
      input_feed[model.dec_inputs[l].name] = dec_inputs[l]
      input_feed[model.dec_masks[l].name] = dec_masks[l]

    last_target = model.dec_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([model.batch_size], dtype=np.int32)

    if not forward_only:
        output_feed = [model.updated[bucket_id],  # Update Op that does SGD.
                     model.gradient_norms[bucket_id],  # Gradient norm.
                     model.losses[bucket_id]]
    else:
        output_feed = [model.losses[bucket_id]]  # Loss for this batch.
        for l in xrange(decoder_size):  # Output logits.
            output_feed.append(model.outputs[bucket_id][l])
    outputs = session.run(output_feed, input_feed)

    if not forward_only:
        return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
        return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

def train():
    buckets,buckets_scale = get_buckets()
    chat_model = ChappieModel(False,config.batch_size)
    chat_model.build_graph()

    saver = tf.train.Saver()
    initial_step = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        restore_saved_parameters(sess, saver)
        iteration = chat_model.global_step.eval()
        total_loss = 0
        while True:
            skip_step = 100
            bucket_id = get_bucket_id(buckets_scale)
            enc_inputs, dec_inputs, dec_masks = data.gen_batch(buckets[bucket_id],
                                                                           bucket_id,
                                                                           batch_size=config.batch_size)
            start = time.time()
            _, step_loss, _ = step(sess, chat_model, enc_inputs, dec_inputs, dec_masks, bucket_id, False)
            total_loss += step_loss
            iteration += 1

            if iteration % skip_step == 0:
                print('Iter {}: loss {}, time {}'.format(iteration, total_loss/skip_step, time.time() - start))
                saver.save(sess, os.path.join(config.cpt_path, 'chatbot'), global_step=chat_model.global_step)
                sys.stdout.flush()

def main():
    train()
if __name__ == '__main__':
    main()
