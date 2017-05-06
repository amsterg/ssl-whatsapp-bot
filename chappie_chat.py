from yowsup.stacks import  YowStackBuilder
from yowsup.layers.auth import AuthError
from yowsup.layers import YowLayerEvent
from yowsup.layers.network import YowNetworkLayer
from yowsup.env import YowsupEnv
from yowsup.layers.interface                           import YowInterfaceLayer, ProtocolEntityCallback
from yowsup.layers.protocol_messages.protocolentities  import TextMessageProtocolEntity
from yowsup.layers.protocol_receipts.protocolentities  import OutgoingReceiptProtocolEntity
from yowsup.layers.protocol_acks.protocolentities      import OutgoingAckProtocolEntity
import sys
import numpy as np
import tensorflow as tf

from skel import ChappieModel
import config
import data
import os

credentials = ("###-###-####", "*******************************") # replace with your phone and password
class chatLayer(YowInterfaceLayer):
    enc_vocab = data.load_vocab("encoder")
    inv_dec_vocab = data.load_vocab("decoder")

    model = ChappieModel(True, batch_size=1)
    model.build_graph()

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.cpt_path+'/checkpoint'))

    print("Loading parameters for the Chatbot")
    saver.restore(sess, ckpt.model_checkpoint_path)

    max_length = config.buckets[-1][0]

    def receive(self, protocolEntity):
        if protocolEntity.getTag() == "message":
            self.onMessage(protocolEntity)
        #self.toLower(receipt)


    def run_step(self,sess, model, enc_inputs, dec_inputs, dec_masks, bucket_id, forward_only):
        """ Run one step in training.
        @forward_only: boolean value to decide whether a backward path should be created
        forward_only is set to True when you just want to evaluate on the test set,
        or when you want to the bot to be in chat mode. """
        encoder_size, decoder_size = config.buckets[bucket_id]
        if len(enc_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                            " %d != %d." % (len(enc_inputs), encoder_size))
        if len(dec_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                           " %d != %d." % (len(dec_inputs), decoder_size))
        if len(dec_masks) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                           " %d != %d." % (len(dec_masks), decoder_size))

        # input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for step in range(encoder_size):
            input_feed[model.enc_inputs[step].name] = enc_inputs[step]
        for step in range(decoder_size):
            input_feed[model.dec_inputs[step].name] = dec_inputs[step]
            input_feed[model.dec_masks[step].name] = dec_masks[step]

        last_target = model.dec_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([model.batch_size], dtype=np.int32)

        # output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [model.train_ops[bucket_id],  # update op that does SGD.
                           model.gradient_norms[bucket_id],  # gradient norm.
                           model.losses[bucket_id]]  # loss for this batch.
        else:
            output_feed = [model.losses[bucket_id]]  # loss for this batch.
            for step in range(decoder_size):  # output logits.
                output_feed.append(model.outputs[bucket_id][step])

        outputs = sess.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

    def onMessage(self, messageProtocolEntity):
        #send receipt otherwise we keep receiving the same message over and over

        if True:
            output_file = open("convos.txt", 'a+')
            receipt = OutgoingReceiptProtocolEntity(messageProtocolEntity.getId(), messageProtocolEntity.getFrom(), 'read', messageProtocolEntity.getParticipant())
            #print "Alien: ",messageProtocolEntity.getBody()
            line = messageProtocolEntity.getBody().rstrip()
            output_file.write('HUMAN :: ' + line + '\n')
            #with open("convos.txt",'aw') as fil:
            #    fil.write("Human:: ",line)

            token_ids = data.sentence2id(str(line))
            if (len(token_ids) > self.max_length):
                print('Max length :', self.max_length)

            bucket_id = min([b for b in range(len(config.buckets))
                                    if config.buckets[b][0] >= len(token_ids)])

            enc_inputs, dec_inputs, dec_masks = data.gen_batch([(token_ids, [])],
                                                                            bucket_id,
                                                                            batch_size=1)
            _, _, output_logits = self.run_step(self.sess, self.model, enc_inputs, dec_inputs,
                                           dec_masks, bucket_id, True)

            outputs = [int (np.argmax(logit, axis=1)) for logit in output_logits]

            if config.eos_id in outputs:
                outputs = outputs[:outputs.index(config.eos_id)]

            
            response = data.ids2seq(outputs,"decoder")
            #with open("convos.txt",'aw') as fil:
            #    fil.write("Bot:: ",response)
            output_file.write('BOT :: ' + response + '\n')
            outgoingMessageProtocolEntity = TextMessageProtocolEntity(
                response,
                to = messageProtocolEntity.getFrom())

            self.toLower(receipt)
            self.toLower(outgoingMessageProtocolEntity)
            output_file.close()
    @ProtocolEntityCallback("receipt")
    def onReceipt(self, entity):
        ack = OutgoingAckProtocolEntity(entity.getId(), "receipt", entity.getType(), entity.getFrom())
        self.toLower(ack)

if __name__==  "__main__":
    stackBuilder = YowStackBuilder()
    print ("Booting up Whatsapp client..")
    stack = stackBuilder\
        .pushDefaultLayers(True)\
        .push(chatLayer())\
        .build()

    stack.setCredentials(credentials)
    stack.broadcastEvent(YowLayerEvent(YowNetworkLayer.EVENT_STATE_CONNECT))   #sending the connect signal
    stack.loop() #this is the program mainloop
