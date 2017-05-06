#import tensorflow as tf
import os
import re
import random
import config
import numpy as np
def process_file():
    DATA_DIR = "data/11th/"
    file1 = DATA_DIR+("1.txt")
    with open(file1,'r') as f:
        content = f.readlines()

    content_lines = []

    for o in range(len(content)):

        b = re.sub(r'\(|\{|\<|\[.*\]|\)|\}|\>','',content[o])

        if b != '\n':
            content_lines.append(b)

    convos = []
    for i in range(len(content_lines)):

        n = re.search(r'episode listing',content_lines[i])
        m = re.search(r'related marks are trademarks',content_lines[i])
        k = re.search(r'Airdate|airdate',content_lines[i])
        c = re.search(r'.*:|.* :',content_lines[i])


        if n == None and m == None and k==None and content_lines[i] != '\r\n' and c!=None  :
            t = content_lines[i].replace('\r','')#.replace('\n','<eos>')


            ti = re.search(r'\S:',t)
            if ti != None:

                t = t.replace(':',' :')

            convos += [t]



    return convos

    """
    with open('convos.txt','w') as f:
        for i in range(len(convos)):
            f.write(convos[i])
    """
def seq2seq(convos):
    seq1,seq2 = [],[]
    for i in range(len(convos)-1):
        seq1 += [convos[i][1]]
        seq2 += [convos[i+1][1]]
    assert len(seq1) == len(seq2)

    return seq1,seq2
def tokenizer(line):
    line = re.sub(',',' ,',line)
    line = re.sub('\.',' .',line)
    line = re.sub('!',' !',line)
    line = re.sub('\?',' ?',line)
    line_split = line.lower().split()
    return line_split

def vocab(convos,coding_type):
    vocab = {}
    #with open("vocab",'wb') as f:
    for conv in convos:
        for token in tokenizer(conv):
            if not token in vocab:
                vocab[token] = 0
            vocab[token] +=1
    #vocab_list  = vocab.keys()
    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    if coding_type == "encoder":
        with open("vocab_enc.txt", 'wb') as f:
            f.write('<pad>' + '\n')
            f.write('<unk>' + '\n')
            f.write('<go>' + '\n')
            f.write('<eos>' + '\n')
            index = 4
            for word in sorted_vocab:
                if vocab[word] < config.threshold:
                    with open('config.py', 'ab') as cf:
                        cf.write('vocab_enc = ' + str(index) + '\n')

                f.write(word+"\n")
                index +=1
    elif coding_type == "decoder":
        with open("vocab_dec.txt", 'wb') as f:
            f.write('<pad>' + '\n')
            f.write('<unk>' + '\n')
            f.write('<go>' + '\n')
            f.write('<eos>' + '\n')
            index = 4
            for word in sorted_vocab:
                if vocab[word] < config.threshold:
                    with open('config.py', 'ab') as cf:
                        cf.write('vocab_dec = ' + str(index) + '\n')

                f.write(word+"\n")
                index +=1

def load_vocab(coding_type):
    if (coding_type == "encoder"):
        with open("vocab_enc.txt",'rb') as f:
            vocab_words = f.read().splitlines()
        words_ids = {vocab_words[i]:i for i in range(len(vocab_words))}
        #print words_ids
    elif (coding_type == "decoder"):
        with open("vocab_dec.txt",'rb') as f:
            vocab_words = f.read().splitlines()
        words_ids = {vocab_words[i]:i for i in range(len(vocab_words))}

    return words_ids
def convos2ids(convos,coding_type):
    conv2id =[]
    word2id_enc = load_vocab("encoder")
    #print word2id_enc
    if coding_type == "encoder":
        for conv in convos:
            #print conv
            conv_tokenized = tokenizer(conv)
            conv2id  += [[word2id_enc.get(token) for token in conv_tokenized]]

    elif coding_type == "decoder":
        word2id_dec = load_vocab("decoder")
        for conv in convos:
                conv_tokenized = tokenizer(conv)
                conv2id  += [[word2id_dec.get(token) for token in conv_tokenized]]

    return conv2id
def sentence2id(sentence):
    sentence_tokd = tokenizer(sentence)
    vocab = load_vocab("encoder")
    return [vocab.get(token, vocab['<unk>']) for token in sentence_tokd]

def ids2seq(ids,coding_type):
    if coding_type == "encoder":
        vocab = load_vocab("encoder")

        rev_vocab = {x:y for y,x in vocab.items()}
        # rev_keys = vocab.values()
        # rev_values = vocab.keys()
        # assert len(rev_keys) == len(rev_values)
        # rev_vocab = {}
        # for i in range(len(rev_keys)):
        #     rev_vocab[rev_keys[i]] = rev_values[i]
        seq = [rev_vocab[id] for id in ids]
        return ' '.join([word for word in seq])

    elif coding_type == "decoder":
        vocab = load_vocab("decoder")

        rev_vocab = {x:y for y,x in vocab.items()}
        # rev_keys = vocab.values()
        # rev_values = vocab.keys()
        # assert len(rev_keys) == len(rev_values)
        # rev_vocab = {}
        # for i in range(len(rev_keys)):
        #     rev_vocab[rev_keys[i]] = rev_values[i]
        seq = [rev_vocab[id] for id in ids]
        return ' '.join([word for word in seq])

def file2ids():

    convos = process_file()
    convos_splitted  = [convos[i].split(':') for i in range(len(convos))]

    encoder_seq,decoder_seq = seq2seq(convos_splitted)

    with open("seqs_enc.txt",'wb') as f:
        for seq in encoder_seq:
            f.write(seq)
    with open("seqs_dec.txt",'wb') as f:
        for seq in decoder_seq:
            f.write(seq)
    vocab(encoder_seq,"encoder")
    vocab(decoder_seq,"decoder")

    conv_enc = convos2ids(encoder_seq,"encoder")
    conv_dec = convos2ids(decoder_seq,"decoder")

    with open("encode_ids.txt",'wb') as f:
        i =0
        for enc in conv_enc:
            re_enc = re.sub(r'\[|\]|,','',str(enc))
            f.write(re_enc+"\n")

            if len(re_enc.split(' ')) > i:#max sequence_size
                i = len(re_enc.split(' '))
        #print i

    with open("decode_ids.txt",'wb') as f:
        i = 0
        for dec in conv_dec:
            re_dec = re.sub(r'\[|\]|,','',str(dec))
            f.write("2 "+re_dec+" 3"+"\n")#2-<go>,3-<eos>

            if len(re_dec.split(' ')) > i:#max sequence_size
                i = len(re_dec.split(' '))

        #print i
def bucket_data():
    data_buckets = [[]for _ in config.buckets]
    with open("encode_ids.txt",'rb') as f1:
        with open("decode_ids.txt",'rb') as f2:
            encode_line= f1.readline()
            decode_line = f2.readline()

            while encode_line and decode_line:
                encode_ids = [int(id_) for id_ in encode_line.split()]
                decode_ids = [int(id_) for id_ in decode_line.split()]
                for id,(enc_size,dec_size) in enumerate(config.buckets):
                    if len(encode_ids) <= enc_size and len(decode_ids) <= dec_size:
                        data_buckets[id].append([encode_ids,decode_ids])
                        break
                encode_line= f1.readline()
                decode_line = f2.readline()
            return data_buckets
def pad_sequence(seq,size):
    seq = seq + [config.pad_id for _ in range(size-len(seq))]
    return seq
def _reshape_batch(inputs, size, batch_size):
    """ Create batch-major inputs. Batch inputs are just re-indexed inputs
    """
    batch_inputs = []
    for length_id in xrange(size):
        batch_inputs.append(np.array([inputs[batch_id][length_id]
                                    for batch_id in xrange(batch_size)], dtype=np.int32))
    return batch_inputs

def gen_batch(data_bucket,bucket_id,batch_size):
    enc_size, dec_size = config.buckets[bucket_id]
    enc_inputs, dec_inputs = [],[]

    for _ in xrange(batch_size):
        enc_input,dec_input = random.choice(data_bucket)
        enc_inputs.append(list(reversed(pad_sequence(enc_input,enc_size))))
        dec_inputs.append(list(pad_sequence(dec_input,dec_size)))
    #print (dec_inputs),len(dec_inputs[0])
    batch_enc_inputs = _reshape_batch(enc_inputs, enc_size, batch_size)
    batch_dec_inputs = _reshape_batch(dec_inputs, dec_size, batch_size)
    #print (batch_dec_inputs)
    batch_masks = []
    for length_id in xrange(dec_size):
        batch_mask = np.ones(batch_size, dtype=np.float32)
        for batch_id in xrange(batch_size):
            # we set mask to 0 if the corresponding target is a PAD symbol.
            # the corresponding decoder is decoder_input shifted by 1 forward.
            if length_id < dec_size - 1:
                target = dec_inputs[batch_id][length_id + 1]
            if length_id == dec_size - 1 or target == config.pad_id:
                batch_mask[batch_id] = 0.0
        batch_masks.append(batch_mask)
    return batch_enc_inputs,batch_dec_inputs,batch_masks

def main():
    file2ids()#vocabulary creation,ids generation,encoder-decoder split.
    data_buckets = bucket_data()
    #print data_buckets
    # #print data_buckets[-1][-1],len(data_buckets[-1][-1][-1])
    #gen_batch(data_buckets[0],0,32)


    #ids2seq(ids)
if __name__ == '__main__':
    main()
