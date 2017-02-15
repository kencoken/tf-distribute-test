import argparse
import math
import tempfile
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

tf_spec = {
    'ps': ['localhost:2222'],
    'worker': ['localhost:2223', 'localhost:2224', 'localhost:2225', 'localhost:2226']
}
sync_replicas = False
learning_rate = 0.01


def model(server):
    worker_device = "/job:%s/task:%d" % (server.server_def.job_name,
                                         server.server_def.task_index)
    is_chief = server.server_def.task_index == 0

    with tf.device(tf.train.replica_device_setter(
            worker_device=worker_device,
            ps_device="/job:ps/cpu:0",
            cluster=tf_spec)):
        global_step = tf.Variable(0, name="global_step", trainable=False)

        fc1_W = tf.Variable(
            tf.truncated_normal([784, 100], stddev=1.0 / 784), name="fc1_W")
        fc1_b = tf.Variable(tf.zeros([100]), name="fc1_b")

        fc2_W = tf.Variable(
            tf.truncated_normal([100, 10], stddev=1.0 / math.sqrt(100)), name="fc2_W")
        fc2_b = tf.Variable(tf.zeros([10]), name="fc2_b")

        # Ops: located on the worker specified with task_index
        x = tf.placeholder(tf.float32, [None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])

        fc1_a = tf.nn.xw_plus_b(x, fc1_W, fc1_b)
        fc1_a_relu = tf.nn.relu(fc1_a)
        y = tf.nn.softmax(tf.nn.xw_plus_b(fc1_a_relu, fc2_W, fc2_b))
        cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

        opt = tf.train.AdamOptimizer(learning_rate)

        if sync_replicas:
            num_workers = len(tf_spec['worker'])

            opt = tf.train.SyncReplicasOptimizer(
                opt,
                replicas_to_aggregate=num_workers,
                total_num_replicas=num_workers,
                name='mnist_sync_replicas')

        train_step = opt.minimize(cross_entropy, global_step=global_step)

        if sync_replicas:
            local_init_op = opt.local_step_init_op
            if is_chief:
                local_init_op = opt.chief_init_op

            ready_for_local_init_op = opt.ready_for_local_init_op

            # Initial token and chief queue runners required by the sync_replicas mode
            chief_queue_runner = opt.get_chief_queue_runner()
            sync_init_op = opt.get_init_tokens_op()

        init_op = tf.global_variables_initializer()
        train_dir = tempfile.mkdtemp()

        if sync_replicas:
            sv = tf.train.Supervisor(
                is_chief=is_chief,
                logdir=train_dir,
                init_op=init_op,
                local_init_op=local_init_op,
                ready_for_local_init_op=ready_for_local_init_op,
                recovery_wait_secs=1,
                global_step=global_step)
        else:
            sv = tf.train.Supervisor(
                is_chief=is_chief,
                logdir=train_dir,
                init_op=init_op,
                recovery_wait_secs=1,
                global_step=global_step)

        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            device_filters=["/job:ps", "/job:worker/task:%d" % server.server_def.task_index])

        # The chief worker (task_index==0) session will prepare the session,
        # while the remaining workers will wait for the preparation to complete.
        if is_chief:
            print("Worker %d: Initializing session..." % server.server_def.task_index)
        else:
            print("Worker %d: Waiting for session to be initialized..." %
                  server.server_def.task_index)

        sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

        if sync_replicas and is_chief:
            # Chief worker will start the chief queue runner and call the init op.
            sess.run(sync_init_op)
            sv.start_queue_runners(sess, [chief_queue_runner])

        return sess, x, y_, train_step, global_step, cross_entropy

def ps_task():
    server = tf.train.Server(tf_spec,
                             job_name='ps',
                             task_index=0)
    server.join()

def worker_task(task_index):
    server = tf.train.Server(tf_spec,
                             job_name='worker',
                             task_index=task_index)

    # Make model
    sess, x, y_, train_step, global_step, _ = model(server)

    # Main loop
    step = 0
    batch_sz = 50
    iters = 55000 / batch_sz
    while step < iters:
        _, step = sess.run([train_step, global_step],
                           feed_dict={x: mnist.train.images[step * batch_sz:(step + 1) * batch_sz],
                                      y_: mnist.train.labels[step * batch_sz:(step + 1) * batch_sz]})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('job_name', help='type of job (ps or worker)', type=str)
    parser.add_argument('task_index', help='number of task', type=int)
    parser.add_argument('--task_count', help='number of tasks', type=int)
    args = parser.parse_args()
    if args.task_count and args.task_count < len(tf_spec['worker']):
        tf_spec['worker'] = tf_spec['worker'][:args.task_count]

    if args.job_name == 'ps':
        ps_task()
    elif args.job_name == 'worker':
        worker_task(args.task_index)
