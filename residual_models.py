import argparse
import pandas as pd
import jack_misc as jack
import tensorflow as tf

def Res_Model(input,seq_lens,mask,ln_mask,dropout,num_outputs,is_train,args):
    #def Model(input,seq_lens,mask,ln_mask,dropout,num_outputs,is_train,args)
    #   
    # all arguments are placeholders except for the class argument args, which
    # can be defined as either an argparse setup like so:
    #
    #
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_res', default=True, type=bool, help='If model is residual or not')
    parser.add_argument('--layout', default="CNN", type=str, help='RNN, CNN, or "CNN RNN" etc')
    parser.add_argument('--RNN_size', default=128, type=int, help='Size of the RNN layer')
    parser.add_argument('--CNN_size', default=64, type=int, help='Size of the CNN layer')
    parser.add_argument('--RNN_depth', default=2, type=int, help='Number of RNN layers')
    parser.add_argument('--CNN_depth', default=20, type=int, help='Number of CNN layers')
    parser.add_argument('--FC_size', default=256, type=int, help='Size of the CNN layers')
    parser.add_argument('--FC_depth', default=1, type=int, help='Number of FC layers')
    parser.add_argument('--activation', default='ELU', type=str, help='Activation function ("ELU","ReLU")')
    parser.add_argument('--bottleneck', default=False, type=bool, help='Bottleneck in LSTM layers')
    parser.add_argument('--filter_dims', default="3", type=str, help='Pattern for filter kx1 dimensions')
    parser.add_argument('--norm_func', default="layer", type=str, help='Which normalisation function to use: layer, batch, none')
    parser.add_argument('--bn_momentum', default=0.99, type=float, help='Batch norm momentum')
    parser.add_argument('--verbose', default=True, type=bool, help='Print stuff')
    args = parser.parse_args()
    '''
    # or a class like so:
    '''
    class args:
        def __init__():
            self.model_res = True
            self.layout = 'CNN RNN' #space-separated CNN RNN arguments, in order
            self.RNN_size = 128
            self.RNN_depth = 2
            self.CNN_depth = 20
            self.CNN_size = 64
            self.FC_size = 256
            self.FC_depth = 2
            self.activation = 'ELU' #ELU or ReLU
            self.bottleneck = True
            self.filter_dims = '3'   #must be a string
            self.norm_func = "layer" # or "batch"
            self.bn_momentum = 0.99 #only relevant if self.norm_func = "batch"
            self.verbse = True
    '''
    #-------------------Model Params
    cnn_size = args.CNN_size
    cnn_depth = args.CNN_depth
    rnn_size = args.RNN_size
    rnn_depth = args.RNN_depth
    fc_size = args.FC_size
    fc_depth = args.FC_depth
    cellfunc = tf.contrib.rnn.LSTMCell 
    if args.activation.lower() == "relu":
        activation_fn = tf.nn.relu 
    elif args.activation.lower() == 'swish':
        activation_fn = jack.swish
    elif args.activation.lower() == 'eswish':
        activation_fn = jack.eswish
    elif args.activation.lower() == 'prelu':
        activation_fn = jack.prelu
    elif args.activation.lower() == 'lrelu':
        activation_fn = tf.nn.leaky_relu
    elif args.activation.lower() == 'softplus':
        activation_fn = tf.nn.softplus
    else:
        activation_fn = tf.nn.elu
    cell_args = {} 
    cell_args.update({'forget_bias': 1.0,'use_peepholes':True})
    layer = [input]
    norm_args = {}
    if args.norm_func.lower() == 'batch':
        norm_func = jack.masked_batch_norm
        norm_args.update({'training':is_train,'decay':args.bn_decay})
        recurrent_norm_func = jack.masked_layer_norm
    elif args.norm_func.lower() == 'layer':
        norm_func = jack.masked_layer_norm
        recurrent_norm_func = jack.masked_layer_norm
    else:
        norm_func = jack.masked_identity
        recurrent_norm_func = jack.masked_identity
    model_layout = args.layout.split()
    for K,k in enumerate(model_layout):
        #-------------------RNN    
        if k == 'RNN':
            if args.model_res == 1 and K>0:
                layer.append(tf.layers.conv1d(layer[-1],rnn_size*2,1,padding='SAME',activation=activation_fn))
                layer.append(norm_func(layer[-1],ln_mask,**norm_args))
            for i in range(rnn_depth):
                with tf.variable_scope('RNN'+str(i)):
                    if args.model_res == 1:
                        res_start_layer = len(layer)-1
                    if args.model_res == 1 and args.bottleneck == True:
                        with tf.variable_scope('bottleneck'+str(i)):
                            layer.append(tf.layers.conv1d(layer[-1],rnn_size,1,padding='SAME',activation=activation_fn))
                            layer.append(norm_func(layer[-1],ln_mask,**norm_args))
                    layer.append(jack.LSTM_layer(layer[-1],cellfunc,cell_args,rnn_size,seq_lens,False))
                    if args.model_res == 1:
                        layer.append(recurrent_norm_func(layer[-1],ln_mask))
                    else:
                        #layer.append(tf.nn.dropout(layer[-1],dropout))
                        layer.append(tf.nn.dropout(recurrent_norm_func(layer[-1],ln_mask),dropout))
                    if args.model_res == 1 and layer[-1].get_shape().as_list()[-1] == layer[res_start_layer].get_shape().as_list()[-1]:
                        with tf.variable_scope('ADDING_LAYER_'+str(res_start_layer)):
                            layer.append(layer[-1]+layer[res_start_layer])
        #-------------------CNN 
        elif k == 'CNN':
            filter_dims_pattern = [int(l) for l in args.filter_dims.split()]
            with tf.variable_scope('initCNN'):
                layer.append(tf.layers.conv1d(layer[-1],cnn_size,filter_dims_pattern[0],padding='SAME',activation=None,bias_initializer=tf.constant_initializer(0.01)))
            for i in range(cnn_depth):
                if args.model_res == 1 and i%2 == 0:
                    res_start_layer = len(layer)-1
                with tf.variable_scope('CNN'+str(i+1)):
                    layer.append(norm_func(activation_fn(layer[-1]),ln_mask,**norm_args))
                    cnn_dim = filter_dims_pattern[i%(len(filter_dims_pattern))]
                    layer.append(tf.layers.conv1d(layer[-1],cnn_size,cnn_dim,padding='SAME',activation=None,bias_initializer=tf.constant_initializer(0.01)))
                    if i%2 == 1 and args.model_res == 1:
                        with tf.variable_scope('ADDING_LAYER_'+str(res_start_layer)):
                            layer.append(layer[-1]+layer[res_start_layer])
                    #else:
                    #    with tf.variable_scope('dropout_test'):
                    #        layer.append(tf.nn.dropout(layer[-1],1-dropout/2))
            with tf.variable_scope('Output_conv_activation'):
                    layer.append(norm_func(activation_fn(layer[-1]),ln_mask,**norm_args))
    #-------------------MASK
    layer.append(tf.boolean_mask(layer[-1],mask))
    #-------------------FC
    for i in range(fc_depth):
        with tf.variable_scope('FC'+str(i)):
            layer.append(jack.FC_layer(layer[-1],fc_size,dropout,activation_fn=activation_fn))
    #-------------------OUTPUT
    with tf.variable_scope('Output'):
        layer.append(jack.FC_layer(layer[-1],num_outputs,1,activation_fn=tf.identity))
    #-------------------SUMMARY    
    for I,i in enumerate(layer):
        print i.get_shape().as_list(),
        print("%i:"%(I)),
        print str(i.name)
    return layer

def Dense_Model(input,seq_lens,mask,ln_mask,dropout,num_outputs,is_train,args):
    #def Model(input,seq_lens,mask,ln_mask,dropout,num_outputs,is_train,args)
    #   
    # all arguments are placeholders except for the class argument args, which
    # can be defined as either an argparse setup like so:
    #
    #
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_res', default=True, type=bool, help='If model is residual or not')
    parser.add_argument('--layout', default="CNN", type=str, help='RNN, CNN, or "CNN RNN" etc')
    parser.add_argument('--RNN_size', default=128, type=int, help='Size of the RNN layer')
    parser.add_argument('--CNN_size', default=64, type=int, help='Size of the CNN layer')
    parser.add_argument('--RNN_depth', default=2, type=int, help='Number of RNN layers')
    parser.add_argument('--CNN_depth', default=20, type=int, help='Number of CNN layers')
    parser.add_argument('--FC_size', default=256, type=int, help='Size of the CNN layers')
    parser.add_argument('--FC_depth', default=1, type=int, help='Number of FC layers')
    parser.add_argument('--activation', default='ELU', type=str, help='Activation function ("ELU","ReLU")')
    parser.add_argument('--bottleneck', default=False, type=bool, help='Bottleneck in LSTM layers')
    parser.add_argument('--filter_dims', default="3", type=str, help='Pattern for filter kx1 dimensions')
    parser.add_argument('--norm_func', default="layer", type=str, help='Which normalisation function to use: layer, batch, none')
    parser.add_argument('--bn_momentum', default=0.99, type=float, help='Batch norm momentum')
    parser.add_argument('--verbose', default=True, type=bool, help='Print stuff')
    args = parser.parse_args()
    '''
    # or a class like so:
    '''
    class args:
        def __init__():
            self.model_res = True
            self.layout = 'CNN RNN' #space-separated CNN RNN arguments, in order
            self.RNN_size = 128
            self.RNN_depth = 2
            self.CNN_depth = 20
            self.CNN_size = 64
            self.FC_size = 256
            self.FC_depth = 2
            self.activation = 'ELU' #ELU or ReLU
            self.bottleneck = True
            self.filter_dims = '3'   #must be a string
            self.norm_func = "layer" # or "batch"
            self.bn_momentum = 0.99 #only relevant if self.norm_func = "batch"
            self.verbse = True
    '''
    #-------------------Model Params
    cnn_size = args.CNN_size
    cnn_depth = args.CNN_depth
    rnn_size = args.RNN_size
    rnn_depth = args.RNN_depth
    fc_size = args.FC_size
    fc_depth = args.FC_depth
    cellfunc = tf.contrib.rnn.LSTMCell 
    if args.activation.lower() == "relu":
        activation_fn = tf.nn.relu 
    elif args.activation.lower() == 'swish':
        activation_fn = jack.swish
    elif args.activation.lower() == 'eswish':
        activation_fn = jack.eswish
    elif args.activation.lower() == 'prelu':
        activation_fn = jack.prelu
    elif args.activation.lower() == 'elu':
        activation_fn = tf.nn.elu
    elif args.activation.lower() == 'softplus':
        activation_fn = tf.nn.softplus
    else:
        activation_fn = tf.nn.leaky_relu
    cell_args = {} 
    filter_dims_pattern = [int(l) for l in args.filter_dims.split()]
    cell_args.update({'forget_bias': 1.0,'use_peepholes':True})
    layer = [input]
    norm_args = {}
    if args.norm_func.lower() == 'batch':
        norm_func = jack.masked_batch_norm
        norm_args.update({'training':is_train,'decay':args.bn_decay})
    elif args.norm_func.lower() == 'layer':
        norm_func = jack.masked_layer_norm
    else:
        norm_func = tf.identity
    model_layout = args.layout.split()
    with tf.variable_scope('Init_conv'):        
        layer.append(jack.conv1d_layer(layer[-1],cnn_size,filter_dims_pattern[0],norm_func,mask,norm_args,1,activation_fn=activation_fn,conv_act_norm_drop=[0]))
    for i in range(cnn_depth):
        with tf.variable_scope('dense_'+str(i)):  
            bottleneck = jack.conv1d_layer(tf.concat([layer[J] for J in range(len(layer)-1,max(len(layer)-args.max_dense,0),-1)],2), cnn_size*4, 1, norm_func, mask, norm_args, 1, activation_fn=activation_fn,conv_act_norm_drop=[1,2,0])
            conv_out = jack.conv1d_layer(bottleneck, cnn_size, filter_dims_pattern[0], norm_func, mask, norm_args, 1, activation_fn=activation_fn,conv_act_norm_drop=[1,2,0])
        layer.append(conv_out)
    with tf.variable_scope('dense_out'):  
        layer.append(jack.conv1d_layer(layer[-1], 0, 0, norm_func, mask, norm_args, 1, activation_fn=activation_fn,conv_act_norm_drop=[1,2]))
    #-------------------MASK
    layer.append(tf.boolean_mask(layer[-1],mask))
    #-------------------FC
    for i in range(fc_depth):
        with tf.variable_scope('FC'+str(i)):
            layer.append(jack.FC_layer(layer[-1],fc_size,dropout,activation_fn=activation_fn))
    #-------------------OUTPUT
    with tf.variable_scope('Output'):
        layer.append(jack.FC_layer(layer[-1],num_outputs,1,activation_fn=tf.identity))
    #-------------------SUMMARY    
    for I,i in enumerate(layer):
        print i.get_shape().as_list(),
        print("%i:"%(I)),
        print str(i.name)
    return layer







def inception_block(input,cnn_size,activation_fn,norm_func,mask,norm_args,blocks,ph_dropout,filter_size=3,residual=False):
    layer = [input]
    if residual==True:
        conv_act_norm_drop = [1,2,0] #activation -> normalize -> dropout -> convolution
        #layer = [jack.conv1d_layer(input,cnn_size*2,1,norm_func,mask,norm_args,1,conv_act_norm_drop=[0])]
    else:
        conv_act_norm_drop = [0,1,2] #convolution -> activation -> normalize -> dropout 
    for I,i in enumerate(blocks):
        outputs = []
        for j in i: # for each path 
            sequence = [layer[-1]]
            for k in range(j): #go through each path
                with tf.variable_scope(str(I)+'.'+str(j)+'.'+str(k)+'inception'):
                    cnn_dim = 1 if k == 0 else filter_size
                    #if k==j-1 and residual==True:
                    #    sequence.append(jack.conv1d_layer(sequence[-1],cnn_size,cnn_dim,norm_func,mask,norm_args,1,activation_fn=activation_fn,conv_act_norm_drop=conv_act_norm_drop)) #activation -> normalize -> convolution
                    #else:
                    sequence.append(jack.conv1d_layer(sequence[-1],cnn_size,cnn_dim,norm_func,mask,norm_args,1-ph_dropout/2, activation_fn=activation_fn, conv_act_norm_drop=conv_act_norm_drop)) # keep prob! so make it 0.75 instead of 0.25, also dropout can happen on the last layer due to the linear convolution following
                    #else:
                    #    
            outputs.append(sequence[-1])       
        with tf.variable_scope(str(I)+'inception_output'):    
            if residual==True:
                concat_out = tf.concat(outputs,2)
                #if concat_out.get_shape()[-1] != layer[-1].get_shape()[-1]:
                with tf.variable_scope(str(I)+'_reshape_layer'):
                    adding_layer = jack.conv1d_layer(concat_out,layer[-1].get_shape()[-1],1,norm_func,mask,norm_args,1,activation_fn=activation_fn, conv_act_norm_drop=conv_act_norm_drop)
                #adding_layer *= 0.2
                #else:
                #    adding_layer = concat_out
                #if I==len(blocks)-1:
                #    layer.append(jack.conv1d_layer(adding_layer + layer[-1],0,1,norm_func,mask,norm_args,1,activation_fn=activation_fn,conv_act_norm_drop=[1,2]))  
                #else:
                #layer.append(jack.conv1d_layer(adding_layer+layer[-1],0,1,norm_func,mask,norm_args,1,activation_fn=activation_fn,conv_act_norm_drop=[1,2]))
                layer.append(adding_layer+layer[-1])
            else:
                if I==len(blocks)-1 and I!=0:
                    outputs.append(jack.conv1d_layer(input,cnn_size,1,norm_func,mask,norm_args,1,activation_fn=activation_fn,conv_act_norm_drop=conv_act_norm_drop)) #skip connection of the network input
                layer.append(tf.concat(outputs,2))
    return layer[-1]

def Inc_Res_Model(input,seq_lens,mask,ln_mask,dropout,num_outputs,is_train,args):
    #def Model(input,seq_lens,mask,ln_mask,dropout,num_outputs,is_train,args)
    #   
    # all arguments are placeholders except for the class argument args, which
    # can be defined as either an argparse setup like so:
    #
    #
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_res', default=True, type=bool, help='If model is residual or not')
    parser.add_argument('--layout', default="CNN", type=str, help='RNN, CNN, or "CNN RNN" etc')
    parser.add_argument('--RNN_size', default=128, type=int, help='Size of the RNN layer')
    parser.add_argument('--CNN_size', default=64, type=int, help='Size of the CNN layer')
    parser.add_argument('--RNN_depth', default=2, type=int, help='Number of RNN layers')
    parser.add_argument('--CNN_depth', default=20, type=int, help='Number of CNN layers')
    parser.add_argument('--FC_size', default=256, type=int, help='Size of the CNN layers')
    parser.add_argument('--FC_depth', default=1, type=int, help='Number of FC layers')
    parser.add_argument('--activation', default='ELU', type=str, help='Activation function ("ELU","ReLU")')
    parser.add_argument('--bottleneck', default=False, type=bool, help='Bottleneck in LSTM layers')
    parser.add_argument('--filter_dims', default="3", type=str, help='Pattern for filter kx1 dimensions')
    parser.add_argument('--norm_func', default="layer", type=str, help='Which normalisation function to use: layer, batch, none')
    parser.add_argument('--bn_momentum', default=0.99, type=float, help='Batch norm momentum')
    parser.add_argument('--verbose', default=True, type=bool, help='Print stuff')
    args = parser.parse_args()
    '''
    # or a class like so:
    '''
    class args:
        def __init__():
            self.model_res = True
            self.layout = 'CNN RNN' #space-separated CNN RNN arguments, in order
            self.RNN_size = 128
            self.RNN_depth = 2
            self.CNN_depth = 20
            self.CNN_size = 64
            self.FC_size = 256
            self.FC_depth = 2
            self.activation = 'ELU' #ELU or ReLU
            self.bottleneck = True
            self.filter_dims = '3'   #must be a string
            self.norm_func = "layer" # or "batch"
            self.bn_momentum = 0.99 #only relevant if self.norm_func = "batch"
            self.verbse = True
    '''
    #-------------------Model Params
    cnn_size = args.CNN_size
    cnn_depth = args.CNN_depth
    rnn_size = args.RNN_size
    rnn_depth = args.RNN_depth
    fc_size = args.FC_size
    fc_depth = args.FC_depth
    cellfunc = tf.contrib.rnn.LSTMCell 
    inception_blocks = [[1,2,3],[1,3]]
    if args.activation.lower() == "relu":
        activation_fn = tf.nn.relu 
    elif args.activation.lower() == 'swish':
        activation_fn = jack.swish
    elif args.activation.lower() == 'eswish':
        activation_fn = jack.eswish
    elif args.activation.lower() == 'lrelu':
        activation_fn = tf.nn.leaky_relu
    elif args.activation.lower() == 'softplus':
        activation_fn = tf.nn.softplus
    else:
        activation_fn = tf.nn.elu
    filter_dims_pattern = [int(l) for l in args.filter_dims.split()]
    cell_args = {} 
    cell_args.update({'forget_bias': 1.0,'use_peepholes':True})
    norm_args = {}
    if args.norm_func.lower() == 'batch':
        norm_func = jack.masked_batch_norm
        norm_args.update({'training':is_train,'decay':args.bn_decay})
    elif args.norm_func.lower() == 'layer':
        norm_func = jack.masked_layer_norm
    else:
        norm_func = tf.identity
    layer=[input]
    if args.model_res==1:
        with tf.variable_scope('init_conv'):
            layer.append(jack.conv1d_layer(layer[-1],cnn_size*2,1,norm_func,mask,norm_args,1,activation_fn=activation_fn,conv_act_norm_drop=[0]))
    for i in range(args.num_inc_blocks):
        with tf.variable_scope('inc'+str(i)):
            layer.append(inception_block(layer[-1],cnn_size,activation_fn,norm_func,ln_mask,norm_args,inception_blocks,dropout,filter_size=filter_dims_pattern[0],residual=(args.model_res==1)))
    with tf.variable_scope('final_slide'):
        conv_act_norm = [0,1,2] if args.model_res==0 else [1,2,0,1,2]
        layer.append(jack.conv1d_layer(layer[-1],cnn_size,11,norm_func,mask,norm_args,1,activation_fn=activation_fn,conv_act_norm_drop=conv_act_norm))
        #layer.append(jack.masked_layer_norm(jack.LSTM_layer(layer[-1],cellfunc,cell_args,rnn_size,seq_lens,False),ln_mask))
    #-------------------MASK
    layer.append(tf.boolean_mask(layer[-1],mask))
    #-------------------FC
    for i in range(fc_depth):
        with tf.variable_scope('FC'+str(i)):
            layer.append(jack.FC_layer(layer[-1],fc_size,dropout,activation_fn=activation_fn))
    #-------------------OUTPUT
    with tf.variable_scope('Output'):
        layer.append(jack.FC_layer(layer[-1],num_outputs,1,activation_fn=tf.identity))
    #-------------------SUMMARY    
    for I,i in enumerate(layer):
        print i.get_shape().as_list(),
        print("%i:"%(I)),
        print str(i.name)
    return layer


def ResNext_Model(input,seq_lens,mask,ln_mask,dropout,num_outputs,is_train,args):
    #def Model(input,seq_lens,mask,ln_mask,dropout,num_outputs,is_train,args)
    #   
    # all arguments are placeholders except for the class argument args, which
    # can be defined as either an argparse setup like so:
    #
    #
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_res', default=True, type=bool, help='If model is residual or not')
    parser.add_argument('--layout', default="CNN", type=str, help='RNN, CNN, or "CNN RNN" etc')
    parser.add_argument('--RNN_size', default=128, type=int, help='Size of the RNN layer')
    parser.add_argument('--CNN_size', default=64, type=int, help='Size of the CNN layer')
    parser.add_argument('--RNN_depth', default=2, type=int, help='Number of RNN layers')
    parser.add_argument('--CNN_depth', default=20, type=int, help='Number of CNN layers')
    parser.add_argument('--FC_size', default=256, type=int, help='Size of the CNN layers')
    parser.add_argument('--FC_depth', default=1, type=int, help='Number of FC layers')
    parser.add_argument('--activation', default='ELU', type=str, help='Activation function ("ELU","ReLU")')
    parser.add_argument('--bottleneck', default=False, type=bool, help='Bottleneck in LSTM layers')
    parser.add_argument('--filter_dims', default="3", type=str, help='Pattern for filter kx1 dimensions')
    parser.add_argument('--norm_func', default="layer", type=str, help='Which normalisation function to use: layer, batch, none')
    parser.add_argument('--bn_momentum', default=0.99, type=float, help='Batch norm momentum')
    parser.add_argument('--verbose', default=True, type=bool, help='Print stuff')
    args = parser.parse_args()
    '''
    # or a class like so:
    '''
    class args:
        def __init__():
            self.model_res = True
            self.layout = 'CNN RNN' #space-separated CNN RNN arguments, in order
            self.RNN_size = 128
            self.RNN_depth = 2
            self.CNN_depth = 20
            self.CNN_size = 64
            self.FC_size = 256
            self.FC_depth = 2
            self.activation = 'ELU' #ELU or ReLU
            self.bottleneck = True
            self.filter_dims = '3'   #must be a string
            self.norm_func = "layer" # or "batch"
            self.bn_momentum = 0.99 #only relevant if self.norm_func = "batch"
            self.verbse = True
    '''
    #-------------------Model Params
    cnn_size = args.CNN_size
    cnn_depth = args.CNN_depth
    rnn_size = args.RNN_size
    rnn_depth = args.RNN_depth
    fc_size = args.FC_size
    fc_depth = args.FC_depth
    cellfunc = tf.contrib.rnn.LSTMCell 
    inception_blocks = [[1,2,4],[1,2]]
    if args.activation.lower() == "relu":
        activation_fn = tf.nn.relu 
    elif args.activation.lower() == 'swish':
        activation_fn = jack.swish
    elif args.activation.lower() == 'eswish':
        activation_fn = jack.eswish
    elif args.activation.lower() == 'elu':
        activation_fn = tf.nn.elu
    elif args.activation.lower() == 'softplus':
        activation_fn = tf.nn.softplus
    else:
        activation_fn = tf.nn.leaky_relu
    filter_dims_pattern = [int(l) for l in args.filter_dims.split()]
    cell_args = {} 
    cell_args.update({'forget_bias': 1.0,'use_peepholes':True})
    norm_args = {}
    if args.norm_func.lower() == 'batch':
        norm_func = jack.masked_batch_norm
        norm_args.update({'training':is_train,'decay':args.bn_decay})
    elif args.norm_func.lower() == 'layer':
        norm_func = jack.masked_layer_norm
    else:
        norm_func = tf.identity
    layer=[input]
    with tf.variable_scope('init_conv'):
        layer.append(jack.conv1d_layer(layer[-1],cnn_size,1,norm_func,mask,norm_args,1,activation_fn=activation_fn,conv_act_norm_drop=[0]))
    for i in range(args.next_blocks):
        blocks = []
        for j in range(args.next_cardinality):
            with tf.variable_scope('bottleneck_'+str(i)+'_'+str(j)):
                path = jack.conv1d_layer(layer[-1],args.next_conv_size,1,norm_func,mask,norm_args,1,activation_fn=activation_fn,conv_act_norm_drop=[1,2,0])
            with tf.variable_scope('conv'+str(i)+'_'+str(j)):
                blocks.append(jack.conv1d_layer(path,args.next_conv_size,3,norm_func,mask,norm_args,1,activation_fn=activation_fn,conv_act_norm_drop=[1,2,0]))
        print(len(blocks))
        with tf.variable_scope('concat_block'+str(i)):
            concat_block = jack.conv1d_layer(tf.concat(blocks,2),cnn_size,1,norm_func,mask,norm_args,1,activation_fn=activation_fn,conv_act_norm_drop=[1,2,0])
        layer.append(concat_block + layer[-1])
    with tf.variable_scope('out_norm_act'):
        layer.append(jack.conv1d_layer(layer[-1],cnn_size,1,norm_func,mask,norm_args,1,activation_fn=activation_fn,conv_act_norm_drop=[1,2]))
    #-------------------MASK
    layer.append(tf.boolean_mask(layer[-1],mask))
    #-------------------FC
    for i in range(fc_depth):
        with tf.variable_scope('FC'+str(i)):
            layer.append(jack.FC_layer(layer[-1],fc_size,dropout,activation_fn=activation_fn))
    #-------------------OUTPUT
    with tf.variable_scope('Output'):
        layer.append(jack.FC_layer(layer[-1],num_outputs,1,activation_fn=tf.identity))
    #-------------------SUMMARY    
    for I,i in enumerate(layer):
        print i.get_shape().as_list(),
        print("%i:"%(I)),
        print str(i.name)
    return layer
