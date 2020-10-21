from tensorflow.keras.layers import *
import pickle as pkl
import numpy as np
from tensorflow.keras.models import Model
from tqdm import tqdm
import tensorflow as tf
from wer import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

Embedding_dim = 300
unit_dim = 256
BATCH_SIZE = 128
Epochs = 20

with open('./preprocessed_data/train_text_input.pkl', 'rb') as f:
    train_input=pkl.load(f)
with open('./preprocessed_data/train_text_output.pkl', 'rb') as f:
    train_output=pkl.load(f)
with open('./preprocessed_data/val_text_input.pkl', 'rb') as f:
    val_input=pkl.load(f)
with open('./preprocessed_data/val_text_output.pkl', 'rb') as f:
    val_output=pkl.load(f)
with open('./preprocessed_data/test_text_input.pkl', 'rb') as f:
    test_input=pkl.load(f)
with open('./preprocessed_data/test_text_output.pkl', 'rb') as f:
    test_output=pkl.load(f)
with open('./preprocessed_data/train_voice_input.pkl', 'rb') as f:
    train_voice=pkl.load(f)
with open('./preprocessed_data/val_voice_input.pkl', 'rb') as f:
    val_voice=pkl.load(f)
with open('./preprocessed_data/test_voice_input.pkl', 'rb') as f:
    test_voice=pkl.load(f)
with open('./preprocessed_data/inp_num2char.pkl', 'rb') as f:
    inp_num2char=pkl.load(f)
with open('./preprocessed_data/inp_char2num.pkl', 'rb') as f:
    inp_char2num=pkl.load(f)
with open('./preprocessed_data/oup_num2char.pkl', 'rb') as f:
    oup_num2char=pkl.load(f)
with open('./preprocessed_data/oup_char2num.pkl', 'rb') as f:
    oup_char2num=pkl.load(f)

print('train size :', len(train_input), len(train_output))
print('val size : ', len(val_input), len(val_output))
print('\n\n', train_input[0],'\n')
print(len(inp_num2char))
print(len(inp_char2num))
print(len(oup_num2char))
print(len(oup_char2num))
class simple_vgg(tf.keras.layers.Layer):
    def __init__(self, dropout_rate = 0.1, hidden_size = 300):
        super(simple_vgg, self).__init__()
        self.conv1_1 = tf.keras.layers.Conv2D(64, (3,3), strides = 1, padding = 'same', activation = 'relu')
        self.conv1_2 = tf.keras.layers.Conv2D(64, (3,3), strides=1, padding = 'same', activation = 'relu')
        self.conv2_1 = tf.keras.layers.Conv2D(128, (3, 3), strides=1, padding = 'same', activation = 'relu')
        self.conv2_2 = tf.keras.layers.Conv2D(128, (3, 3), strides=1, padding = 'same', activation = 'relu')
        self.conv3_1 = tf.keras.layers.Conv2D(256, (3, 3), strides=1, padding = 'same', activation = 'relu')
        self.conv3_2 = tf.keras.layers.Conv2D(256, (3, 3), strides=1, padding = 'same', activation = 'relu')
        self.conv4_1 = tf.keras.layers.Conv2D(512, (3, 3), strides=1, padding = 'same', activation = 'relu')
        self.conv4_2 = tf.keras.layers.Conv2D(512, (3, 3), strides=1, padding = 'same', activation = 'relu')
        self.max_pooling = tf.keras.layers.MaxPool2D()
        self.dropout = tf.keras.layers.Dropout(rate = dropout_rate)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(hidden_size, activation = 'linear')
        
    def vgg_calculate(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.max_pooling(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.max_pooling(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.max_pooling(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.max_pooling(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x
    def call(self, x):
        vgg_output = []
        for x_i in range(x.shape[1]):
            vgg_output.append(self.vgg_calculate(x[:,x_i]))
        vgg_output = tf.stack(vgg_output, axis = 1)
        return vgg_output

class Encoder(Model):
    def __init__(self, input_vocab_size, embedding_dim, hidden_size, dropout_rate = 0.1):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.token_embedding = Embedding(input_vocab_size, embedding_dim)
        self.bi_lstm_text = Bidirectional(LSTM(hidden_size, 
                                               dropout = dropout_rate, 
                                               return_sequences = True, 
                                               return_state = True,
                                               recurrent_initializer = 'glorot_uniform'))
        self.bi_lstm_voice = Bidirectional(LSTM(hidden_size, 
                                                dropout = dropout_rate, 
                                                return_sequences = True, 
                                                recurrent_initializer = 'glorot_uniform'))
        self.layernorm = LayerNormalization()
        self.vgg = simple_vgg()

    def call(self, input_text, input_voice, initial_state):
        input_text = self.token_embedding(input_text)
        input_voice = tf.expand_dims(input_voice, axis = -1)
        input_voice = self.vgg(input_voice)

        input_text, fh_state, fc_state, bh_state, bc_state = self.bi_lstm_text(input_text, initial_state = initial_state)
        hidden_state = Concatenate()([fh_state, bh_state])
        cell_state = Concatenate()([fc_state, bc_state])
        state = [hidden_state, cell_state]

        input_voice = self.bi_lstm_voice(input_voice, initial_state = initial_state)

        input_text = self.layernorm(input_text)
        input_voice = self.layernorm(input_voice)

        return input_text, input_voice, state

    def initial_state(self, batch_size):
        hidden  = tf.zeros(shape = (batch_size, self.hidden_size), dtype = tf.dtypes.float32)        
        return [hidden, hidden, hidden, hidden]

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self,units):
        super(BahdanauAttention,self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query,values):
        #query => encoder hidden
        #values => decoder hidden
        hidden_with_time_axis = tf.expand_dims(query, 1)
        # hidden_with_time_axis의 shape은 (batch_size, 1, hidden_size)이다.
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        # self.W1(values) => (batch_size, seq_len, units)
        # self.W2(hidden_with_time_axis) => (batch_size, 1, units)

        # score shape => (batch_size, seq_len, 1)
        attention_weights = tf.nn.softmax(score, axis = 1)
        # softmax encoder의 단어의 중요도를 각각 얻기 위해 사용한다. 
        context_vector = attention_weights*values
        context_vector = tf.reduce_sum(context_vector, axis = 1)
        
        return context_vector, attention_weights


class Decoder(Model):
    def __init__(self, output_vocab_size, embedding_dim, hidden_size, dropout_rate):
        super(Decoder, self).__init__()
        self.target_embedding = Embedding(output_vocab_size, embedding_dim)
        self.concat = Concatenate()
        self.lstm = LSTM(hidden_size, dropout = dropout_rate,return_state = True, return_sequences = True, recurrent_initializer = 'glorot_uniform')
        self.weight_matrix = Dense(output_vocab_size)
        self.voice_attention = BahdanauAttention(hidden_size)
        self.text_attention = BahdanauAttention(hidden_size)

    def call(self, input_token, enc_text, enc_voice, hidden_state):
        # input_token = tf.expand_dims(input_token, 1)
        input_token = self.target_embedding(input_token)
        text_contvec, text_attention_weights = self.text_attention(hidden_state[0], enc_text)
        voice_contvec, voice_attention_weights = self.voice_attention(hidden_state[0], enc_voice)

        text_contvec = tf.expand_dims(text_contvec, 1)
        voice_contvec = tf.expand_dims(voice_contvec, 1)

        lstm_input = self.concat([text_contvec, input_token])

        output, state_h, state_c = self.lstm(lstm_input, initial_state = hidden_state)

        state = [state_h, state_c]

        output = self.weight_matrix(output + text_contvec + voice_contvec)
        return output, state, text_attention_weights, voice_attention_weights


encoder = Encoder(len(inp_char2num), Embedding_dim, unit_dim//2, dropout_rate = 0.1)
decoder = Decoder(len(oup_char2num), Embedding_dim, unit_dim, dropout_rate = 0.1)


optimizer = tf.keras.optimizers.Adam()
# 최적화 함수 정의
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = 'none')

def loss_function(real,pred):
    mask = tf.math.logical_not(tf.math.equal(real,0))
    loss_ = loss_object(real,pred)
    
    mask = tf.cast(mask,dtype = loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

def train_step(text_inp, voice_inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_text, enc_voice, enc_hidden = encoder(text_inp, voice_inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([oup_char2num['<s>']] * BATCH_SIZE, 1)

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _, _ = decoder(dec_input, enc_text, enc_voice, dec_hidden)
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss

def inp_make_sentences(sentences):
    word_sents =  []
    for sentence in sentences:
        word_sent = []
        for word in sentence:
            if int(word) != 0:
                word_sent.append(inp_num2char[word])
        word_sents.append(word_sent)
    return word_sents


def oup_make_sentences(sentences):
    word_sents =  []
    for sentence in sentences:
        word_sent = []
        for word in sentence:
            if int(word) not in [oup_char2num['<s>'],oup_char2num['<e>'], oup_char2num['<p>']]:
                word_sent.append(oup_num2char[word])
            if int(word) == oup_char2num['<e>']:
                break
        word_sents.append(word_sent)
    return word_sents

def val_step(text_inp, voice_inp,  targ, enc_hidden):
    enc_text, enc_voice, enc_hidden = encoder(text_inp, voice_inp, enc_hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([oup_char2num['<s>']] * BATCH_SIZE, 1)
    step_predictions = []

    for t in range(1, targ.shape[1]):
        predictions, dec_hidden, _, _ = decoder(dec_input, enc_text, enc_voice, dec_hidden)
        predictions = tf.argmax(predictions, axis = -1)
        predictions = tf.reshape(predictions, [predictions.shape[0], 1])
        step_predictions.append(predictions.numpy())
        dec_input = predictions

    step_predictions = np.concatenate(step_predictions, axis = -1)
    step_predictions = oup_make_sentences(step_predictions)
    targ = oup_make_sentences(targ)
    step_wer = 0
    for tar_sent, pred_sent in zip(targ, step_predictions):
        step_wer += wer(tar_sent, pred_sent)
    step_wer = step_wer // BATCH_SIZE

    return step_wer

steps_per_epoch = len(train_input)//BATCH_SIZE
val_steps_per_epoch = len(val_input)//BATCH_SIZE
for epoch in tqdm(range(Epochs)):
    enc_hidden = encoder.initial_state(BATCH_SIZE)
    total_loss = 0
    val_wer = 0
    for batch in tqdm(range(steps_per_epoch)):
        text_batch_input = train_input[batch * BATCH_SIZE : (batch+1) * BATCH_SIZE]
        voice_batch_input = train_voice[batch * BATCH_SIZE : (batch+1) * BATCH_SIZE]
        target = train_output[batch * BATCH_SIZE : (batch+1) * BATCH_SIZE]
        batch_loss = train_step(text_batch_input, voice_batch_input, target, enc_hidden)
        total_loss += batch_loss.numpy()

    for batch in tqdm(range(val_steps_per_epoch)):

        text_batch_input = val_input[batch * BATCH_SIZE : (batch+1) * BATCH_SIZE]
        voice_batch_input = val_voice[batch * BATCH_SIZE : (batch+1) * BATCH_SIZE]
        target = val_output[batch * BATCH_SIZE : (batch+1) * BATCH_SIZE]
        
        batch_wer = val_step(text_batch_input, voice_batch_input, target, enc_hidden)
        val_wer += batch_wer
        
    totaL_loss = total_loss / steps_per_epoch
    val_wer = val_wer / val_steps_per_epoch

    print('Epoch {}  Loss {:.4f}'.format(epoch + 1, total_loss))
    print('validation wer {:.4f}'.format(val_wer))


def test_step(text_inp, target, enc_hidden):
    enc_text, enc_hidden = encoder(text_inp, enc_hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([oup_char2num['<s>']] * 1, 1)
    step_predictions = []
    text_att_weights = []
    voice_att_weights = []
    for t in range(1, 37):
        predictions, dec_hidden, text_attention_weights, voice_attention_weights = decoder(dec_input, enc_text, dec_hidden)
        text_att_weights.append(text_attention_weights)
        voice_att_weights.append(voice_attention_weights)

        predictions = tf.argmax(predictions, axis = -1)
        predictions = tf.reshape(predictions, [predictions.shape[0], 1])
        step_predictions.append(predictions.numpy())
        dec_input = predictions

    step_predictions = np.concatenate(step_predictions, axis = -1)
    step_predictions = oup_make_sentences(step_predictions)
    target = oup_make_sentences([target])
    step_wer = wer(step_predictions[0], target[0])
    return step_predictions, step_wer, text_

wer_score = 0
for src_sent, tar_sent in tqdm(zip(test_input, test_output)):
    wer_score += test_step(np.expand_dims(src_sent, 0), tar_sent, encoder.initial_state(1))[1]

print(wer_score/len(test_input))