from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import colorlog
import time
import numpy as np
import pandas as pd
from utils.data_utils import enqueue
from utils.configuration import ModelConfig
from model.model import CSMN
from scripts.generate_dataset import EOS_ID
from utils.evaluator import Evaluator
from termcolor import colored
flags = tf.app.flags

flags.DEFINE_string('eval_dir', './checkpoints/eval',
                           """Directory where to write event logs.""")
flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
flags.DEFINE_string("train_dir", "./checkpoints", "checkpoint directory [checkpoints]")
flags.DEFINE_string(
    "vocab_fname",
    "./data_yfcc/hashtag_dataset/40000.vocab",
    "Vocabulary file for evaluation"
)
flags.DEFINE_integer("num_gpus", 1, "Number of gpus to use")
flags.DEFINE_integer('eval_interval_secs', 60 * 1,
                            """How often to run the eval.""")
flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")

flags.DEFINE_float("hold", 0.5, "Threshold")

flags.DEFINE_integer('DF_num', 0, " ")

TOWER_NAME = 'tower'



FLAGS = flags.FLAGS

def _load_vocabulary(vocab_fname):
  with open(vocab_fname, 'r') as f:
    vocab = f.readlines()
  vocab = [s.strip() for s in vocab]
  rev_vocab = {}
  for i, token in enumerate(vocab):
    rev_vocab[i] = token
  return vocab, rev_vocab

def _inject_summary(key_value):
    summary = tf.Summary()
    for key, value in key_value.iteritems():
      summary.value.add(tag='%s' % (key), simple_value=value)
    return summary

def _eval_once(saver, summary_writer, argmaxs, answer_ids, argmaxs2, answer_ids2,probs2, vocab, rev_vocab,
    num_examples_per_epoch, b_global_step,tower_img):
  """Run Eval once.
  """
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)) as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)

      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      if global_step == b_global_step:
          return global_step
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))
      num_iter = 1 + int(
          num_examples_per_epoch / FLAGS.batch_size / FLAGS.num_gpus
      )


      desc_list = []
      answer_list = []
      desc_token_list = []
      answer_token_list = []
      desc_list2 = []
      answer_list2 = []
      prob_list2 = []
      name_list = []
      SP0=[]
      SP1=[]
      SP2=[]
      AS_P0=[]
      AS_P1=[]
      AS_P2=[]

      step = 0
      cor = 0
      cor2 = 0

      while step < num_iter and not coord.should_stop():
        results = sess.run([argmaxs, answer_ids,argmaxs2, answer_ids2, probs2,tower_img])
        desc = results[0].tolist()
        answer = results[1].tolist()
        desc2 = results[2].tolist()
        answer2 = results[3].tolist()
        SM_prob2 = results[4].tolist()

        np_prob2 = np.array(SM_prob2)
        
        St_np_prob=np.sort(np_prob2, axis=1)
        AS_np_prob= np.argsort(np_prob2, axis=1)
        name_=results[5][0]
        desc_list += desc
        answer_list += answer
        desc_list2 += desc2
        answer_list2 += answer2
        prob_list2 += SM_prob2
        name_list.extend(name_)

        SP0.extend(St_np_prob[:,-1])
        SP1.extend(St_np_prob[:,-2])
        SP2.extend(St_np_prob[:,-3])
        AS_P0.extend(AS_np_prob[:,-1])
        AS_P1.extend(AS_np_prob[:,-2])
        AS_P2.extend(AS_np_prob[:,-3])
        step += 1
        #if step==10:        print(answer)

      sigTH=0.5  
      wh_prob = np.where(np.array(prob_list2) >= sigTH )
      F1_prob = np.zeros([len(prob_list2),len(prob_list2[0])])
      F1_prob[wh_prob] = 1

      DFOut=pd.DataFrame()
      DFOut['GT']= answer_list2
      DFOut['All_prob']= prob_list2
      DFOut['First']= AS_P0
      DFOut['Prob1']= SP0
      DFOut['Second']= AS_P1
      DFOut['Prob2']= SP1
      DFOut['Third']= AS_P2
      DFOut['Prob3']= SP2
      DFOut['Name']= name_list
    

      DFOut.to_excel(str(FLAGS.eval_dir)+"/ALL_Output.xlsx", index=False)
      print(str(ckpt.model_checkpoint_path))
      for i in xrange(len(desc_list)):
        desc = []
        answer = []
        for k in xrange(len(desc_list[i])):
          token_id = desc_list[i][k]
          if token_id == EOS_ID:
            break
          desc.append(rev_vocab[token_id])
        for k in xrange(len(answer_list[i])):
          token_id = answer_list[i][k]
          if token_id == EOS_ID:
            break
          answer.append(rev_vocab[token_id])
        desc_token_list.append(desc)
        answer_token_list.append(answer)

      TP = 0
      FP = 0
      FN = 0
      TN = 0

      for X in range(len(desc_list2)):
        for Y in range(20):
          if (int(answer_list2[X][Y]==1) and F1_prob[X][Y]==1):
            TP+=1
          elif (int(answer_list2[X][Y]==0) and F1_prob[X][Y]==1):
            FP+=1
          elif (int(answer_list2[X][Y]==1) and F1_prob[X][Y]==0):
            FN+=1
          else: TN+=1

      print('\nTP, FP, FN, TN=', TP, FP, FN, TN)

      precision=(TP/(TP+FP))
      recall=(TP/(TP+FN))
      F1SCORE= 2*((precision*recall)/(precision+recall))
      Acc = (TP+TN)/(TP+ FP+ FN+ TN)

      print('Confusion Matrix: Acc, Pre, Re, F1 = ', Acc, precision,recall,F1SCORE)

      cor_list=[]
      asls=[]
      predl=[]
      cor_list2=[]
      predl2=[]
      ab=[]
      nm=[]
      allpred=[]
      for agmx in range(len(desc_list2)):
        if (int(answer_list2[agmx][int(AS_P0[agmx])])==1) or (int(answer_list2[agmx][int(AS_P1[agmx])])==1):
            cor2+=1
            asls.append(answer_list2[agmx])
            cor_list.append(AS_P0[agmx])
            predl.append(SP0[agmx])
            cor_list2.append(AS_P1[agmx])
            predl2.append(SP1[agmx])
            nm.append(name_list[agmx])
            allpred.append(F1_prob[agmx])
            if (int(answer_list2[agmx][int(AS_P0[agmx])])==1) and(int(answer_list2[agmx][int(AS_P1[agmx])])==1):
              ab.append('oth')
            elif (int(answer_list2[agmx][int(AS_P0[agmx])])==1):
              ab.append('Main')
            elif (int(answer_list2[agmx][int(AS_P1[agmx])])==1):
              ab.append('Sub')
            #print("AGMX",desc_list2[agmx], answer_list2[agmx], prob_list2[agmx][desc_list2[agmx]])

        if (int(answer_list2[agmx][int(AS_P0[agmx])])==1):
            cor+=1

      print("\nFisrt category prediction: ", cor/len(desc_list2))  
      print("Fisrt and Second category prediction: ", cor2/len(desc_list2))      

      DFcor=pd.DataFrame()
      DFcor['GT']= asls
      DFcor['Final pred']= allpred
      DFcor['Main']= cor_list
      DFcor['Bigest prob']= predl
      DFcor['Sub']= cor_list2
      DFcor['Second prob']= predl2
      DFcor['What is Correct']= ab
      DFcor['Name']= nm
      DFcor.to_excel(str(FLAGS.eval_dir)+"/Main_category_prediction.xlsx", index=False)


    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
  return global_step



def evaluate():
  # Read vocabulary
  vocab, rev_vocab = _load_vocabulary(FLAGS.vocab_fname)

  with tf.Graph().as_default() as g:
    #Enque data for evaluation
    num_examples_per_epoch, tower_img_embedding, tower_context_length, \
        tower_caption_length, tower_context_id, tower_caption_id, \
        tower_answer_id, tower_answer_new,tower_context_mask, \
        tower_caption_mask, name_list = enqueue(True)

    tower_argmax = []
    tower_argmax2 = []
    tower_prob2 = []
    tower_img=[]
    
    # Calculate the gradients for each model tower.
    with tf.variable_scope(tf.get_variable_scope()) as scope:
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
            inputs = [
                tower_img_embedding[i],
                tower_context_length[i],
                tower_caption_length[i],
                tower_context_id[i],
                tower_caption_id[i],
                tower_answer_id[i],
                tower_answer_new[i],
                tower_context_mask[i],
                tower_caption_mask[i]
            ]
            net = CSMN(inputs, ModelConfig(FLAGS), is_training= False)
            argmax = net.argmax
            argmax2 = net.argmax2
            prob2 = net.prob2
            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()

            # Keep track of the gradients across all towers.
            tower_argmax.append(argmax)
            tower_argmax2.append(argmax2)
            tower_prob2.append(prob2)
            tower_img.append(name_list[i])

    argmaxs = tf.concat(tower_argmax, 0)
    argmaxs2 = tf.concat(tower_argmax2, 0)
    probs2 = tf.concat(tower_prob2, 0) 

    answer_ids = tf.concat(tower_answer_id, 0)

    answer_ids2 = tf.concat(tower_answer_new, 0)
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    #Don't evaluate again for the same checkpoint.
    b_g_s = "0"
    while True:
      c_g_s = _eval_once(
          saver, summary_writer, argmaxs, answer_ids, argmaxs2, answer_ids2, probs2,vocab,
          rev_vocab, num_examples_per_epoch, b_g_s,tower_img
      )
      b_g_s = c_g_s
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):
  if not tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
