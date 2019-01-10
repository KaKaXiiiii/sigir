import tensorflow as tf
import numpy as np

class SiameseLSTM(object):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses an character embedding layer, followed by a biLSTM and Energy Loss layer.
    """
    
    def BiRNN(self, x, dropout, scope, embedding_size, sequence_length):
        n_input=embedding_size
        n_steps=sequence_length
        n_hidden=n_steps
        n_layers=3
        # Prepare data shape to match `bidirectional_rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input) (?, seq_len, embedding_size)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # Reshape to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])
        print(x)
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(x, n_steps, 0)
        print(x)
        # Define lstm cells with tensorflow
        # Forward direction cell
        with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope):
            stacked_rnn_fw = []
            for _ in range(n_layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,output_keep_prob=dropout)
                stacked_rnn_fw.append(lstm_fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

        with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
            stacked_rnn_bw = []
            for _ in range(n_layers):
                bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell,output_keep_prob=dropout)
                stacked_rnn_bw.append(lstm_bw_cell)
            lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)
        # Get lstm cell output

        with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
            outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)
        #return outputs[-1]

        # output transformation to the original tensor type
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])
        return outputs
    
    def contrastive_loss(self, y,d,batch_size):
        tmp= y *tf.square(d)
        #tmp= tf.mul(y,tf.square(d))
        tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
        return tf.reduce_sum(tmp +tmp2)/batch_size/2

    # return 1 output of lstm cells after pooling, lstm_out(batch, step, rnn_size * 2)
    def max_pooling(self, lstm_out):
        height, width = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])       # (step, length of input for one step)

        # do max-pooling to change the (sequence_lenght) tensor to 1-lenght tensor
        lstm_out = tf.expand_dims(lstm_out, -1)
        output = tf.nn.max_pool(
            lstm_out,
            ksize=[1, height, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID')

        output = tf.reshape(output, [-1, width])

        return output

    def avg_pooling(self, lstm_out):
        height, width = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])       # (step, length of input for one step)
        
        # do avg-pooling to change the (sequence_lenght) tensor to 1-lenght tensor
        lstm_out = tf.expand_dims(lstm_out, -1)
        output = tf.nn.avg_pool(
            lstm_out,
            ksize=[1, height, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID')
        
        output = tf.reshape(output, [-1, width])
        
        return output

    def attentive(self, input_q, input_a, att_W):
        h_q, w = int(input_q.get_shape()[1]), int(input_q.get_shape()[2])
        h_a = int(input_a.get_shape()[1])

        #output_q = tf.reduce_mean(input_q, axis=1)
        output_q = self.avg_pooling(input_q)
        #output_q = self.max_pooling(input_q)

        reshape_q = tf.expand_dims(output_q, 1)
        reshape_q = tf.tile(reshape_q, [1, h_a, 1])
        reshape_q = tf.reshape(reshape_q, [-1, w])
        reshape_a = tf.reshape(input_a, [-1, w])

        M = tf.tanh(tf.add(tf.matmul(reshape_q, att_W['Wqm']), tf.matmul(reshape_a, att_W['Wam'])))
        M = tf.matmul(M, att_W['Wms'])

        S = tf.reshape(M, [-1, h_a])
        S = tf.nn.softmax(S)

        S_diag = tf.matrix_diag(S)
        attention_a = tf.matmul(S_diag, input_a)
        attention_a = tf.reshape(attention_a, [-1, h_a, w])

        #output_a = tf.reduce_mean(attention_a, axis=1)
        output_a = self.avg_pooling(attention_a)
        #output_a = self.max_pooling(attention_a)

        return tf.tanh(output_q), tf.tanh(output_a)
    
    def __init__(
        self, sequence_length, vocab_size, embedding_size, hidden_units, l2_reg_lambda, batch_size, embedding_matrix, mode):

        # Placeholders for input, output and dropout
        self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
        self.input_y = tf.placeholder(tf.int64, [None], name="input_y")
        self.add_fea = tf.placeholder(tf.float32, [None, 4], name="add_fea")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0, name="l2_loss")
          
        # Embedding layer
        with tf.name_scope("embedding"):
            '''
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                trainable=True,name="W")
            '''
            # char embedding
            if embedding_matrix.all() != None:
                self.W = tf.Variable(embedding_matrix, trainable=False, name="emb", dtype=tf.float32)
            else:
                self.W = tf.get_variable("emb", [self.num_chars, self.emb_dim])
            
            self.embedded_chars1 = tf.nn.embedding_lookup(self.W, self.input_x1)
            #self.embedded_chars_expanded1 = tf.expand_dims(self.embedded_chars1, -1)
            self.embedded_chars2 = tf.nn.embedding_lookup(self.W, self.input_x2)
            #self.embedded_chars_expanded2 = tf.expand_dims(self.embedded_chars2, -1)
        #'''
        attention_size = 2*sequence_length
        with tf.name_scope("att_weight"):
            self.att_W = {
                'Wam' : tf.Variable(tf.random_uniform([2*sequence_length,attention_size], -0.1, 0.1), trainable=True),
                'Wqm' : tf.Variable(tf.random_uniform([2*sequence_length,attention_size], -0.1, 0.1), trainable=True),
                'Wms' : tf.Variable(tf.random_uniform([attention_size,1], -0.1, 0.1), trainable=True)
            }
        #'''
        # Create a convolution + maxpool layer for each filter size
        with tf.name_scope("output"):
            self.h1 = self.BiRNN(self.embedded_chars1, self.dropout_keep_prob, "side1", embedding_size, sequence_length)
            self.h2 = self.BiRNN(self.embedded_chars2, self.dropout_keep_prob, "side2", embedding_size, sequence_length)
            #self.out1 = self.avg_pooling(self.h1)
            #self.out2 = self.avg_pooling(self.h2)

            self.out1, self.out2 = self.attentive(self.h1, self.h2, self.att_W)
            self.mix = tf.reshape(tf.reduce_sum(tf.multiply(self.out1,self.out2),axis=1),[-1,1])
            if mode == 'raw':
                self.outputs = tf.concat([self.out1, self.mix, self.out2],1)
                self.softmax_w = tf.get_variable("softmax_w", [4*sequence_length+1, 2])
            else:
                self.outputs = tf.concat([self.out1, self.mix, self.out2, self.add_fea],1)
                self.softmax_w = tf.get_variable("softmax_w", [4*sequence_length+5, 2])
            self.softmax_b = tf.get_variable("softmax_b", [2])
            self.prob = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b
            self.soft_prob = tf.nn.softmax(self.prob, name='distance')
            self.predictions = tf.argmax(tf.nn.softmax(self.prob), 1, name="predictions")

            
        with tf.name_scope("loss"):
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.prob, labels=tf.one_hot(self.input_y,2))
            self.loss = tf.reduce_sum(self.cross_entropy)

            self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(0.0001),weights_list=tf.trainable_variables())

            self.total_loss = self.loss + self.l2_loss
        #### Accuracy computation is outside of this class.
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
