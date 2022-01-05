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
    x1 = tf.keras.layers.Dense(256, activation="relu")(x1)

    x2_in = tf.keras.layers.Input(shape=(dim,))
    x2 = tf.keras.layers.Embedding(input_dim=21,output_dim=64)(x2_in)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128,return_sequences=False))(x2)

    x = tf.keras.layers.Concatenate()([x1,x2])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256,activation="relu",name="outx")(x)
    x = tf.keras.layers.Dense(128,activation="relu")(x)
    x_out = tf.keras.layers.Dense(1,activation="sigmoid")(x)
    model = tf.keras.Model(inputs=[x1_in,x2_in],outputs=[x_out])
    return model


pool = Pool(cpu_count())

p250_path = "../data/ACPs250.txt"
n250_path = "../data/non-ACPs250.txt"

p82_path = "../data/ACPs82.txt"
n82_path = "../data/non-ACPs82.txt"

p740_path = "../data/acp740_p.fa"
n740_path = "../data/acp740_n.fa"

in_path = "./data/ACPs10.txt"

p_seqs = list(read_data(p82_path).values())
n_seqs = list(read_data(n82_path).values())
#
# p_seqs = list(read_data(n740_path).values())
# n_seqs = list(read_data(p740_path).values())
#
test_p_fs = np.array(pool.map(get_features,p_seqs))
test_n_fs = np.array(pool.map(get_features,n_seqs))

test_p_emb = np.array(pool.map(embed,p_seqs))
test_n_emb = np.array(pool.map(embed,n_seqs))

test_fs = np.concatenate([test_p_fs,test_n_fs])
test_emb = np.concatenate([test_p_emb,test_n_emb])

test_label = np.array([1]*len(test_p_fs) + [0]*len(test_n_fs))

def scheduler(epoch, lr):
  if epoch < 15:
    return lr
  else:
    return lr * tf.math.exp(-0.5)

estop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
lrs = tf.keras.callbacks.LearningRateScheduler(scheduler)

model_path = "/home/ys/ex1/work/app/code/ACP.model_0.878048"
model_path = "ACP.model"
model = tf.keras.models.load_model(model_path)
res = model.predict([test_fs,test_emb])
label = []

t = 0.5
for x in res:
    if x>t:
        label.append(1)
    else:
        label.append(0)

tn, fp, fn, tp = metrics.confusion_matrix(test_label, label).ravel()
precise = metrics.precision_score(test_label,label)
acc = metrics.accuracy_score(test_label,label)
f1 = metrics.f1_score(test_label,label)
recall = metrics.recall_score(test_label,label)
mcc = metrics.matthews_corrcoef(test_label,label)
auc = metrics.roc_auc_score(test_label,res)
ap = metrics.average_precision_score(test_label,res)

cmp = '''
对比论文中的
PRE 86.5
ACC 82.9
F1 82.0
recall 78.0
MCC 0.662

'''
sn = tp/(tp+fn)
sp = tn/(fp+tn)
print(cmp)

print("my model")
print("sn",sn)
print("sp",sp)

print("tp",tp)
print("tn",tn)
print("fn",fn)
print("fp",fp)

print("acc ",acc)
print("f1 ",f1)
print("recall ",recall)
print("precise ",precise)
print("mcc ",mcc)
print("auc ",auc)
print("AP",ap)


'''
n = 30

sn 0.8780487804878049
sp 0.9146341463414634
tp 72
tn 75
fn 10
fp 7
acc  0.8963414634146342
f1  0.8944099378881988
recall  0.8780487804878049
precise  0.9113924050632911
mcc  0.7932139586608681
auc  0.9445270672218917
AP 0.9601905006779669
'''

print("another data predict")

path = "../data/ACPs10.txt"
seq_dict = read_data(path)
ids = list(seq_dict.keys())
seqs = list(seq_dict.values())

fs = []
onehots = []
for k,seq in seq_dict.items():
    vec = get_features(seq)
    onehot = embed(seq)

    fs.append(vec)
    onehots.append(onehot)

fs = np.array(fs)
onehots = np.array(onehots)

res = model.predict([fs,onehots])[:,0]
# t = np.mean(res)
t = 0.5
labels = [0 if x<t else 1 for x in res ]

print(res)
for i in range(len(res)):
    print(i,res[i],labels[i],ids[i],seqs[i])



