import tensorflow as tf
#import numpy as np
import time
def gen_batches(num_batches):
    DATA = "data/heart.csv"
    BATCH_SIZE = 3
    file_queue = tf.train.string_input_producer([DATA])
    reader = tf.TextLineReader(skip_header_lines=1)
    key,value = reader.read(file_queue)
    record_defaults = [[1.0] for _ in range(9)]
    record_defaults.append([1])

    cols = tf.decode_csv(value,record_defaults = record_defaults)
    features = tf.stack(cols[:-1])

    min_after_dequeue = 10 * BATCH_SIZE

    capacity = 20 * BATCH_SIZE
    data_batch,label_batch = tf.train.shuffle_batch([features,cols[-1]],batch_size = BATCH_SIZE,capacity = capacity,
                                            min_after_dequeue = min_after_dequeue)
    label_batch = tf.reshape(label_batch,shape = [3,1])
    return data_batch,label_batch

    """
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)

        for _ in range(num_batches):
            ex,label = sess.run([data_batch,label_batch])
            #print ex,label
            #print "#########"
        coord.request_stop()
        coord.join(threads)
    #with open(data,'rb') as file:
        #lines = file.readlines()

    #names = lines[0].split()
    """

def reg(data_batch,label_batch):
    lr = 0.01
    with tf.name_scope(name = "placeholders"):
        X = tf.placeholder(dtype = tf.float32,shape = [3,9],name =  "examples")
        Y = tf.placeholder(dtype = tf.int32,shape = [3,1],name = "labels")

    W = tf.Variable(tf.random_normal(shape = [9,1],stddev = 0.1),name = 'weights')
    b = tf.Variable(tf.zeros([1,1]),name = "bias")

    with tf.name_scope("logits"):
        logits = tf.matmul(X,W)+b


    with tf.name_scope("loss"):
        entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = Y,name = "loss")
        loss = tf.reduce_mean(entropy)
        tf.summary.scalar('loss',loss)
    with tf.name_scope("optimizer"):
        optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    merged = tf.summary.merge_all()
    with tf.Session() as sess:
        total_loss = 0
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        #start_time = time.time()
        init = tf.global_variables_initializer()
        sess.run(init)
        #print sess.run(logits)
        #print "bef"
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        for i in range(154):

            ex_batch,la_batch = sess.run([data_batch,label_batch])

            _,loss_batch,logits_batch,summary,entr = sess.run([optimizer,loss,logits,merged,entropy],feed_dict = {X:ex_batch,Y:la_batch})
            total_loss += loss_batch
            print logits_batch,entr
            print 'losses at step {0}: batch_loss{1}: total_loss{2}'.format(i, loss_batch,total_loss)
            writer.add_summary(summary,global_step = i)
        coord.request_stop()
        coord.join(threads)
        writer.flush()
        writer.close()


def main():
    data_batch,label_batch = gen_batches(154)
    print data_batch,label_batch
    reg(data_batch,label_batch)
if __name__ == '__main__':
    main()
