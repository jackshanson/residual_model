import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.python.training import moving_averages

class bcolors:
    RED   = "\033[1;31m"  
    BLUE  = "\033[1;34m"
    CYAN  = "\033[1;36m"
    GREEN = "\033[0;32m"
    RESET = "\033[0;0m"
    BOLD    = "\033[;1m"
    REVERSE = "\033[;7m"

#-------------------------------------------------------------------------------
#
#       FILE PROCESSING
#
#-------------------------------------------------------------------------------
def get_pccp_dic(fname):
    with open(fname,'r') as f:
        pccp = f.read().splitlines()
        pccp = [i.split() for i in pccp]
        pccp_dic = {i[0]: np.array(i[1:]).astype(float) for i in pccp}
    return pccp_dic

def read_pccp(fname,seq,pccp_dic):
    return np.array([pccp_dic[i] for i in seq])

def read_pssm(fname,seq):
    num_pssm_cols = 44
    pssm_col_names = [str(j) for j in range(num_pssm_cols)]
    with open(fname,'r') as f:
        tmp_pssm = pd.read_csv(f,delim_whitespace=True,names=pssm_col_names).loc[:,'2':'21'].dropna().values[2:,:].astype(float) #'inf' value sometimes nan...?
    if tmp_pssm.shape[0] != len(seq):
        raise ValueError('PSSM file is in wrong format or incorrect!')
    return tmp_pssm

def read_hhm(fname,seq):
    num_hhm_cols = 22
    hhm_col_names = [str(j) for j in range(num_hhm_cols)]
    with open(fname,'r') as f:
        hhm = pd.read_csv(f,delim_whitespace=True,names=hhm_col_names)
    pos1 = (hhm['0']=='HMM').idxmax()+3
    num_cols = len(hhm.columns)
    hhm = hhm[pos1:-1].values[:,:num_hhm_cols].reshape([-1,44])
    hhm[hhm=='*']='9999'
    if hhm.shape[0] != len(seq):
        raise ValueError('HHM file is in wrong format or incorrect!')
    return hhm[:,2:-12].astype(float)    

def spd3_feature_sincos(x,seq,norm_ASA=True):
    ASA = x[:,0]
    if norm_ASA==True:
        rnam1_std = "ACDEFGHIKLMNPQRSTVWY-X"
        ASA_std = (115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                            185, 160, 145, 180, 225, 115, 140, 155, 255, 230,1,1)
        dict_rnam1_ASA = dict(zip(rnam1_std, ASA_std))
        ASA_div =  np.array([dict_rnam1_ASA[i] for i in seq])
        ASA = (ASA/ASA_div)[:,None]
    else:
        ASA = ASA[:,None]
    angles = x[:,1:5]
    HSEa = x[:,5:7]
    HCEprob = x[:,7:10]
    angles = np.deg2rad(angles)
    angles = np.concatenate([np.sin(angles),np.cos(angles)],1)
    return np.concatenate([ASA,angles,HSEa,HCEprob],1)

#USE THIS TO READ SPIDER3 FILES
def read_spd33_output(fname,seq):
    with open(fname,'r') as f:
        spd3_features = pd.read_csv(f,delim_whitespace=True).values[:,3:].astype(float)
    tmp_spd3 = spd3_feature_sincos(spd3_features,seq)
    if tmp_spd3.shape[0] != len(seq):
        raise ValueError('Spider3 file is in wrong format or incorrect!')
    return tmp_spd3

#Non standard SPIDER-3 functions
def read_spd33_third_iteration(fname,seq):
    with open(fname,'r') as f:
        spd3_features = pd.read_csv(f,delim_whitespace=True,skiprows=1,header=None).values[:,1:].astype(float)
    tmp_spd3 = spd3_feature_sincos(spd3_features,seq,norm_ASA=False)
    if tmp_spd3.shape[0] != len(seq):
        raise ValueError('Spider3 file is in wrong format or incorrect!')
    return tmp_spd3
#Non standard SPIDER-3 functions
def read_spd33_features(fnameclass,fnamereg,seq):
    with open(fnameclass,'r') as f:
        spd3_class = pd.read_csv(f,delim_whitespace=True,header=None,skiprows=1).values.astype(float)
    with open(fnamereg,'r') as f:
        spd3_reg = pd.read_csv(f,delim_whitespace=True,header=None,skiprows=1).values.astype(float)[:,:-3]
    theta_tau_phi_psi = spd3_reg[:,1:9]*2-1 #convert sigmoided outputs to sin/cos outputs
    #switch tt and pp around
    spd3_reg[:,1:9] = np.concatenate([theta_tau_phi_psi[:,4:],theta_tau_phi_psi[:,:4]],1)
    spd3_reg[:,9] *= 50 
    spd3_reg[:,10] *= 65 
    spd3_features = np.concatenate([spd3_reg,spd3_class],1)
    if spd3_features.shape[0] != len(seq):
        raise ValueError('Spider3 file is in wrong format or incorrect!')
    return spd3_features

def tee(fname,in_str,append=True):
    fperm = 'a' if append==True else 'w'
    with open(fname,fperm) as f:
        print(in_str),  
        f.write(in_str)

def read_omega_file(fname):
    with open(fname,'r') as f:
        contents = f.read().splitlines()
    AA = contents[1]
    omega = np.array(contents[3].split()).astype(float)
    return AA, omega

def read_fasta_file(fname):
    with open(fname,'r') as f:
        AA = f.read().splitlines()[1]
    return AA

def read_seq_file(fname):
    with open(fname,'r') as f:
        AA = f.read()
    return AA

def read_disorder_file(fname,seq):
    with open(fname,'r') as f:
        do = np.array([i for i in f.read().splitlines()[0]])
    if do.shape[0] != len(seq):
        raise ValueError('Disorder file and sequence different lengths!')
    do_ret = np.ones(do.shape)*-1
    do_ret[do=='O'] = 0
    do_ret[do=='-'] = 0
    do_ret[do=='0'] = 0
    do_ret[do=='D'] = 1 #some files use D
    do_ret[do=='+'] = 1 #some files use +
    do_ret[do=='1'] = 1 #some files use 1  -> MobiDB
    return do_ret

#-------------------------------------------------------------------------------
#
#       MODEL FUNCTIONS
#
#-------------------------------------------------------------------------------

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tfsigmoid(x):
    return 1/(1+tf.exp(-x))

def softmax(x,axis=1):
    return np.exp(x)/(np.sum(np.exp(x),1)[:,None])

def swish(x,B=1):
    B = tf.get_variable('beta',shape=[1],initializer=tf.constant_initializer(1.))
    return x*tf.sigmoid(x*B)

def eswish(x,b=1.25):
    b = tf.get_variable('beta',shape=[1],initializer=tf.constant_initializer(1.))
    return b*x*tf.sigmoid(x)

def prelu(_x, scope=None):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=[1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

def conv1d_layer(input,cnn_size,cnn_width,norm_func,mask,norm_args,ph_dropout,activation_fn=tf.nn.relu,conv_act_norm_drop=[0,1,2,3]):
    for I,i in enumerate(conv_act_norm_drop):
        if i == 0:
            with tf.variable_scope('conv_op_'+str(I)):
                input = tf.layers.conv1d(input,cnn_size,cnn_width,padding='SAME',activation=None,kernel_initializer=tf.orthogonal_initializer(),bias_initializer=tf.constant_initializer(0.01))
        elif i == 1:
            with tf.variable_scope('conv_op_'+str(I)):
                input = activation_fn(input)
        elif i == 2:
            with tf.variable_scope('conv_op_'+str(I)):
                input=norm_func(input,mask,**norm_args)
        elif i == 3:
            with tf.variable_scope('conv_op_'+str(I)):
                input=tf.nn.dropout(input,ph_dropout)
    return input

def LSTM_layer(input,cell,cell_args,cellsize,seq_lens,time_major):
    cells = [cell(cellsize,**cell_args),cell(cellsize,**cell_args)]
    (fw,bw),_ = tf.nn.bidirectional_dynamic_rnn(cells[0],cells[1],input,sequence_length=seq_lens,dtype=tf.float32,swap_memory=True,time_major=time_major)
    return tf.concat([fw,bw],2)

def FC_layer(input,netsize,dropout,activation_fn=tf.nn.relu):
    W = tf.get_variable('W',shape=[input.get_shape().as_list()[-1],netsize],initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
    b = tf.get_variable('b',shape=[netsize],initializer=tf.constant_initializer(0.01))
    output = tf.nn.dropout(activation_fn(tf.matmul(input,W)+b),dropout)
    return output

def get_masked_mean_var(input,mask,axes,scaler,keep_dims=True):
    num = tf.reduce_sum(tf.multiply(input,mask),axis=axes, keep_dims=keep_dims)
    den = tf.multiply(tf.reduce_sum(mask,axis=axes,keep_dims=keep_dims),scaler)#has expand dims on 2nd dimension
    mean = tf.divide(num,den)
    num = tf.reduce_sum(tf.multiply(tf.square(input-mean),mask),axis=axes, keep_dims=keep_dims)
    var = tf.divide(num,den)
    return mean,var

def masked_layer_norm(inp,mask,scope=''):
    with tf.variable_scope('Layer_Norm'+scope):
        input_shape = inp.get_shape().as_list()[-1:]
        beta = tf.Variable(tf.constant(0.0, shape=input_shape), trainable=True,name='beta')
        gamma = tf.Variable(tf.constant(1.0, shape=input_shape), trainable=True,name='Gamma')
        floatmask = tf.cast(tf.expand_dims(mask,2),tf.float32)
        mean,var = get_masked_mean_var(inp,floatmask,[1,2],input_shape[-1])
        variance_epsilon = 1e-12        
        normed = tf.nn.batch_normalization(
            inp, mean, var,offset=beta,scale=gamma,variance_epsilon=variance_epsilon)
        masked_normed = tf.multiply(normed,floatmask)
        return masked_normed

def masked_identity(inp,mask,scope=''):
    return inp

def masked_batch_norm(inp,mask,training=False,decay=0.99,scope=''):
    with tf.variable_scope('Jack_Batch_Norm'+scope):
        inputdepth = tf.cast(tf.shape(inp)[-1],tf.float32)
        batchsize = tf.cast(tf.shape(inp)[0],tf.float32)
        floatmask = tf.cast(tf.expand_dims(mask,2),tf.float32)
        input_shape = inp.get_shape().as_list()[-1:]

        #moving_mean = tf.Variable(tf.constant(0.0, shape=[inputdepth]), trainable=False,name='moving_mean')
        moving_mean = tf.get_variable("moving_mean", input_shape, dtype=tf.float32,  initializer=tf.zeros_initializer, trainable=False)
        moving_var = tf.get_variable("moving_var", input_shape, dtype=tf.float32,  initializer=tf.constant_initializer(1), trainable=False)

        #moving_var = tf.Variable(tf.constant(1.0, shape=[inputdepth]), trainable=False,name='moving_var')   

        batch_mean,batch_var = get_masked_mean_var(inp,floatmask,[0,1],1,keep_dims=False)
        #ema = tf.train.ExponentialMovingAverage(decay=decay)   

        def mean_var_with_update():
            #ema_apply_op = ema.apply([batch_mean, batch_var])
            #with tf.control_dependencies([ema_apply_op]):
            #    return tf.identity(batch_mean), tf.identity(batch_var) 
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, batch_mean, decay, zero_debias=True)
            update_moving_variance = moving_averages.assign_moving_average(moving_var, batch_var, decay, zero_debias=False)
            with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(training,
                            true_fn = mean_var_with_update,
                            false_fn = lambda: (moving_mean, moving_var))

        variance_epsilon = 1e-12       
        beta = tf.get_variable("beta", input_shape, dtype=tf.float32,  initializer=tf.constant_initializer(0), trainable=True)
        #beta = tf.Variable(tf.constant(0.0, shape=[inputdepth]), trainable=True,name='Beta')
        gamma = tf.get_variable("gamma", input_shape, dtype=tf.float32,  initializer=tf.constant_initializer(1), trainable=True)
        #gamma = tf.Variable(tf.constant(1.0, shape=[inputdepth]), trainable=True,name='Gamma')       
        normed = tf.nn.batch_normalization(inp, mean, var, beta, gamma, variance_epsilon)
        masked_normed = tf.multiply(normed,floatmask)
        return masked_normed
#-------------------------------------------------------------------------------
#
#       RESULTS ANALYSIS
#
#-------------------------------------------------------------------------------

def sensitivity(tp,tn,fp,fn):
    return tp/(tp+fn).astype(float)

def specificity(tp,tn,fp,fn):
    return tn/(tn+fp).astype(float)

def precision(tp,tn,fp,fn):
    with np.errstate(invalid='ignore'):
        return tp/(tp+fp).astype(float)

def accuracy(tp,tn,fp,fn):
    return (tp+tn)/(tp+tn+fn+fp).astype(float)

def AUC(sens,spec):
    return np.trapz(sens,spec)

def Sw(tp,tn,fp,fn):
    return sensitivity(tp,tn,fp,fn) + specificity(tp,tn,fp,fn) - 1

def MCC(tp,tn,fp,fn):
    with np.errstate(invalid='ignore'):
        return ((tp*tn)-(fp*fn))/np.sqrt((tp+fp)*(tp+fn)*(tn+fn)*(tn+fp).astype(np.float64))
