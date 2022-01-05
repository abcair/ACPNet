import random
import pandas as pd
from Bio import SeqIO
from propy.PseudoAAC import GetAPseudoAAC
from propy.CTD import CalculateCTD
from sklearn.model_selection import train_test_split
from multiprocessing import Pool,cpu_count
import numpy as np
from sklearn import metrics
from propy.QuasiSequenceOrder import GetQuasiSequenceOrder
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.preprocessing import minmax_scale
import propy

import tensorflow as tf
import tensorflow_addons as tfa

physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

std = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]

dim = 30

def read_data(path):
    records = list(SeqIO.parse(path,format="fasta"))
    res_dict = {}
    for x in records:
        id =  str(x.id)
        seq = str(x.seq)
        res_dict[id]=seq
    return res_dict

def get_features(seq):
    r1 = list(GetAPseudoAAC(seq,lamda=5).values())
    x  = ProteinAnalysis(seq)
    r2 = [x.gravy()]
    r3 = [x.molecular_weight()]
    r4 = list(x.get_amino_acids_percent().values())
    r5 = [x.charge_at_pH(pH=i) for i in range(14)]
    x6 = list(x.secondary_structure_fraction())
    res = r1 + r2 + r3 + r4 + r5 + x6
    return res

def embed(seq):
    mat = [0]*dim
    for i,x in enumerate(seq[0:dim]):
        mat[i] = std.index(x)+1
    return mat

def split_train_val_test(seqs_list):
    train_seqs,val_test_seqs = train_test_split(seqs_list,test_size=0.4,random_state=1,shuffle=True)
    val_seqs,  test_seqs = train_test_split(val_test_seqs,test_size=0.5,random_state=1,shuffle=True)
    return train_seqs,val_seqs,test_seqs

def get_model(dim_fs):
    x1_in = tf.keras.layers.Input(shape=(dim_fs,),name="inx")
    x1 = tf.keras.layers.BatchNormalization()(x1_in)
    x1 = tf.keras.layers.Dense(256, activation="relu",kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5,l2=1e-5))(x1)

    x2_in = tf.keras.layers.Input(shape=(dim,))
    x2 = tf.keras.layers.Embedding(input_dim=21,output_dim=128)(x2_in)
    x2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128,return_sequences=False))(x2)

    x = tf.keras.layers.Concatenate()([x1,x2])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256,activation="relu",name="outx",kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5,l2=1e-5))(x)
    x = tf.keras.layers.Dense(128,activation="relu")(x)
    x_out = tf.keras.layers.Dense(1,activation="sigmoid")(x)
    model = tf.keras.Model(inputs=[x1_in,x2_in],outputs=[x_out])
    return model

def shuffle_seq(seq,num=5):
    np.random.seed(1)
    index = np.random.randint(0,len(seq),num)
    seq_list = list(seq)
    for i in index:
        seq_list[i] = random.choice("ATCG")
    return "".join(seq_list)

def aug(seqs_list,n=10):
    seqs = seqs_list
    seqx = []
    for i in range(n):
        seqs1 = [shuffle_seq(seq,num=int(len(seq)/10)) for seq in seqs]
        seqx.extend(seqs1)
    return seqs + seqx

pool = Pool(cpu_count())

p250_path = "../data/ACPs250.txt"
n250_path = "../data/non-ACPs250.txt"

p_seqs = list(read_data(p250_path).values())
n_seqs = list(read_data(n250_path).values())

p_seqs = aug(p_seqs,n=10)
n_seqs = aug(n_seqs,n=10)

train_p_fs = np.array(pool.map(get_features,p_seqs))
train_n_fs = np.array(pool.map(get_features,n_seqs))

train_p_emb = np.array(pool.map(embed,p_seqs))
train_n_emb = np.array(pool.map(embed,n_seqs))

train_fs = np.concatenate([train_p_fs,train_n_fs])
train_emb = np.concatenate([train_p_emb,train_n_emb])

train_label = np.array([1]*len(train_p_fs) + [0]*len(train_n_fs))

def scheduler(epoch, lr):
  if epoch < 15:
    return lr
  else:
    return lr * tf.math.exp(-0.5)

estop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
lrs = tf.keras.callbacks.LearningRateScheduler(scheduler)

model = get_model(dim_fs=train_fs.shape[1])
print(model.summary())
model.compile(loss=tf.keras.losses.binary_crossentropy,
              # optimizer=tf.keras.optimizers.Adamax(learning_rate=5e-4),
              optimizer=tf.keras.optimizers.Adamax(learning_rate=3e-4),
              metrics=['accuracy',"AUC"])

model.fit(x=[train_fs,train_emb],
          y=train_label,
          batch_size = 16,
          shuffle=True,
          validation_split=0.2,
          verbose=1,
          epochs=5000,
          callbacks=[estop,lrs])

model.save("ACP.model")



