import logging

import colorlog
import tensorflow as tf

from model.model import CSMN
from utils.data_utils import enqueue
from utils.configuration import ModelConfig
from datetime import datetime
import time
import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

flags = tf.app.flags

flags.DEFINE_integer("num_gpus", 2, "Number of gpus to use")
flags.DEFINE_string('train_dir', './checkpoints',
                           """Directory where to write event logs """
                           """and checkpoint.""")
flags.DEFINE_float("init_lr", 0.001, "initial learning rate [0.01]")
flags.DEFINE_float("max_grad_norm", 100, "clip gradients to this norm [100]")
flags.DEFINE_integer("max_steps", 10960, "number of steps to use during training [500000]")
flags.DEFINE_integer('Check', 0,"check")

FLAGS = flags.FLAGS
# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 5.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.8  # Learning rate decay factor.
TOWER_NAME = 'tower'

def _tower_loss(inputs, scope):
  net = CSMN(inputs, ModelConfig(FLAGS))
  loss = net.loss
  loss_new = net.loss2
  tf.summary.scalar(scope+'loss_new', loss_new)
  tf.summary.scalar(scope+'loss', loss)
  Wf2, bf2=net.Wf2,net.bf2
  return loss,loss_new, Wf2, bf2

def _average_gradients(tower_grads):
  """
    From tensorflow cifar 10 tutorial codes
    Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    if(grad_and_vars[0][0])==None: continue
    for g, _ in grad_and_vars:
      if g == None : continue
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train():
    colorlog.basicConfig(
        filename=None,
        level=logging.INFO,
        format="%(log_color)s[%(levelname)s:%(asctime)s]%(reset)s %(message)s",
        datafmt="%Y-%m-%d %H:%M:%S"
    )

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    with tf.Session(
        config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options=gpu_options
        )
    ) as sess:
      global_step = tf.get_variable(
          'global_step', [],
          initializer=tf.constant_initializer(0), trainable=False)
      num_examples_per_epoch, tower_img_embedding, tower_context_length, \
          tower_caption_length, tower_context_id, tower_caption_id, \
          tower_answer_id, tower_answer_new,tower_context_mask, \
          tower_caption_mask = enqueue(False)

      # Calculate the learning rate schedule.
      num_batches_per_epoch = (num_examples_per_epoch /
                               FLAGS.batch_size / FLAGS.num_gpus)
      decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

      # Decay the learning rate exponentially based on the number of steps.
      lr = tf.train.exponential_decay(FLAGS.init_lr,
                                      global_step,
                                      decay_steps,
                                      LEARNING_RATE_DECAY_FACTOR,
                                      staircase=True)

      # Create an optimizer that performs gradient descent.
      opt = tf.train.AdamOptimizer(lr)
      opt2 = tf.train.AdamOptimizer(lr)

      # Calculate the gradients for each model tower.
      tower_grads = []
      tower_grads_new = []
      with tf.variable_scope(tf.get_variable_scope()) as scope:
        for i in xrange(FLAGS.num_gpus):
          with tf.device('/gpu:%d' % i):
            with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
              # Calculate the loss for one tower of the CIFAR model. This function
              # constructs the entire CIFAR model but shares the variables across
              # all towers.
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
              loss, loss_new, Wf2, bf2 = _tower_loss(inputs, scope)

              # Reuse variables for the next tower.
              tf.get_variable_scope().reuse_variables()

              with tf.variable_scope("Wf3"):
                wf3 = tf.get_variable("weights")
                assert wf3.name == "Wf3/weights:0"
                bf3 = tf.get_variable("biases")
                assert bf3.name == "Wf3/biases:0"

              with tf.variable_scope("Wf4"):
                wf4 = tf.get_variable("weights")
                assert wf4.name == "Wf4/weights:0"
                bf4 = tf.get_variable("biases")
                assert bf4.name == "Wf4/biases:0"

              with tf.variable_scope("Wf5"):
                wf5 = tf.get_variable("weights")
                assert wf5.name == "Wf5/weights:0"
                bf5 = tf.get_variable("biases")
                assert bf5.name == "Wf5/biases:0"

              with tf.variable_scope("Wf6"):
                wf6 = tf.get_variable("weights")
                assert wf6.name == "Wf6/weights:0"
                bf6 = tf.get_variable("biases")
                assert bf6.name == "Wf6/biases:0" 

              # Retain the summaries from the final tower.
              summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

              # Calculate the gradients for the batch of data on this CIFAR tower.
              grads = opt.compute_gradients(loss)
              for i in grads:
                  if(i[0]==None):
                    grads.remove(i)
              tower_grads.append(grads)

              grads_new = opt2.compute_gradients(loss_new, var_list = [Wf2,bf2, wf3,bf3,wf4,bf4,wf5,bf5,wf6,bf6])
              for ii in grads_new:
                  if(ii[0]==None):
                    grads_new.remove(ii)    
              tower_grads_new.append(grads_new)

              # Keep track of the gradients across all towers.
              tower_grads.append(grads)

      # We must calculate the mean of each gradient. Note that this is the
      # synchronization point across all towers.
      grads = _average_gradients(tower_grads)
      grads_new= _average_gradients(tower_grads_new)

      # Add a summary to track the learning rate.
      summaries.append(tf.summary.scalar('learning_rate', lr))
      clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], \
          FLAGS.max_grad_norm), gv[1]) for gv in grads]

      clipped_grads_and_vars_new = [(tf.clip_by_norm(gv_new[0], \
          FLAGS.max_grad_norm), gv_new[1]) for gv_new in grads_new]

      # Apply the gradients to adjust the shared variables.
      apply_gradient_op = opt.apply_gradients(
          clipped_grads_and_vars, global_step=global_step
      )

      apply_gradient_op_new = opt2.apply_gradients(
          clipped_grads_and_vars_new, global_step=global_step
      )
      # Create a saver.
      saver = tf.train.Saver(tf.global_variables(), max_to_keep=200)

      # Build the summary operation from the last tower summaries.
      summary_op = tf.summary.merge(summaries)


      # Build an initialization operation to run below.
      init = tf.global_variables_initializer()


      if (FLAGS.Check == 0):   
          sess.run(init)
          ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
          if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
          # Start the queue runners.
          tf.train.start_queue_runners(sess=sess)
          summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
          for step in xrange(FLAGS.max_steps):
            
            start_time = time.time()
            _, loss_value, lr_value = sess.run([apply_gradient_op, loss, lr])

            duration = time.time() - start_time
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            
            if (step + 1)% 10 == 0:
              num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
              examples_per_sec = num_examples_per_step / duration
              sec_per_batch = duration / FLAGS.num_gpus

              format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                            'sec/batch)')
              c_g_step = int(global_step.eval(session=sess))

              print (format_str % (datetime.now(), c_g_step, loss_value,
                                   examples_per_sec, sec_per_batch))
            if (step + 1)% 548 == 0:
              summary_str = sess.run(summary_op)
              summary_writer.add_summary(summary_str, c_g_step)

            # Save the model checkpoint periodically.
            if (step + 1)% 2740 == 0 or (step + 1) == FLAGS.max_steps:
              checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
              saver.save(sess, checkpoint_path, global_step=c_g_step)
          
      elif (FLAGS.Check == 1):   
          sess.run(init)

          ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
          if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
          # Start the queue runners.
          tf.train.start_queue_runners(sess=sess)
          summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
          #print(int(global_step))

          for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value_new, lr_value = sess.run([apply_gradient_op_new, loss_new, lr])
           
            duration = time.time() - start_time

            assert not np.isnan(loss_value_new), 'Model diverged with loss_new = NaN'

            if (step + 1)% 10 == 0:
              num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
              examples_per_sec = num_examples_per_step / duration
              sec_per_batch = duration / FLAGS.num_gpus

              format_str = ('%s: step %d, CEloss = %.2f (%.1f examples/sec; %.3f '
                            'sec/batch)')
              c_g_step = int(global_step.eval(session=sess))
              
              print (format_str % (datetime.now(), c_g_step, loss_value_new,
                                   examples_per_sec, sec_per_batch))

            if (step + 1)% 548 == 0:
              summary_str = sess.run(summary_op)
              summary_writer.add_summary(summary_str, c_g_step)

            # Save the model checkpoint periodically.
            if (step + 1)% 2740 == 0 or (step + 1) == FLAGS.max_steps:
              checkpoint_path = os.path.join(FLAGS.train_dir, 'model2.ckpt')
              saver.save(sess, checkpoint_path, global_step=c_g_step)

def main(argv=None):
  if not tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == "__main__":
    tf.app.run()
