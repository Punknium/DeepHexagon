import directKey
import time
import screen
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from model import model
import math



esc = 0x01
space = 0x39
a = 0x1e
d = 0x20

input_shape = None

# Hyper Params
data_collection_iterations = 200
number_of_actions = 3
r_decay_rate = 0.5
optimizer_learning_rate = 0.001




def resetGame():
    directKey.ReleaseKey(a)
    directKey.ReleaseKey(d)
    time.sleep(0.1)
    directKey.PressKey(esc)
    time.sleep(0.1)
    directKey.ReleaseKey(esc)
    time.sleep(1)
    directKey.PressKey(space)
    time.sleep(0.1)
    directKey.ReleaseKey(space)
    time.sleep(0.5)


def execChoice(choice):
    # print(choice)
    if choice[0] == 1: # Move Left
        directKey.ReleaseKey(d)
        directKey.PressKey(a)
        time.sleep(0.005)
    elif choice[2] == 1: # Move Right
        directKey.ReleaseKey(a)
        directKey.PressKey(d)
        time.sleep(0.005)
    elif choice[1] == 1: # Stop
        directKey.ReleaseKey(a)
        directKey.ReleaseKey(d)
        time.sleep(0.005)
    else:
        print('Invalid Move')


def collect_data(sess, x, y):
    inputs = np.zeros([data_collection_iterations, input_shape[1], input_shape[2], input_shape[3]], dtype=float)
    choices = np.zeros([data_collection_iterations, number_of_actions], dtype=float)
    r_values = np.zeros([data_collection_iterations, 1], dtype=float)

    start_time = time.time()

    for iter in range(0, data_collection_iterations):
        # Grab input and reshape it.
        input = screen.grab_screen()
        input.shape = input_shape
        time.sleep(0.001)
        # print(input[0, 256, 32, 0])
        # Check if the game needs reset
        if input[0, 256, 32, 0] < 10 or input[0, 256, 32, 0] > 250:
            # print('Resetting')
            r = time.time() - start_time
            # print(r)
            start_time = time.time()
            resetGame()
            iter -= 1
            if r_values[iter] == 0:
                r_values[iter] = r
        else:
            choice = np.zeros([number_of_actions])
            # Evaluate Network
            p = sess.run(y, feed_dict={x: input})

            print(p)

            p[0][1] += 0.0001
            p[0] = p[0]/np.sum(p[0])

            # Randomly choose one
            a = np.random.choice(
                a=number_of_actions,
                p=p[0]
            )
            choice[a] = 1

            execChoice(choice)

            # Store values
            inputs[iter] = input
            choices[iter] = choice

    return (inputs, choices, r_values)


def propagate_r_values(r_values):
    new_r_values = np.zeros(r_values.shape)
    r = 0
    d = 1
    idx = r_values.shape[0]
    while idx > 0:
        idx -= 1
        r_value = r_values[idx]
        if r_value[0] != 0:
            r = r_value[0] + 0
            d = 1
        r *= 1/d
        d += r_decay_rate
        new_r_values[idx][0] = r
    return new_r_values


def main():
    # input should be last 4ish frames.
    # Collect Data
    #   Put input into network
    #   Semi-randomly pick an output
    #   Record input, pick, r(None if it didn't change)
    # Propagate r values
    #   Start at end
    #   if r == None:
    #     r = last_r * 1/d
    #   else:
    #     reset
    # Train on Data
    #   prediction = model(input)
    #   loss = prediction * choice * r

    ts = math.floor(time.time())

    input = screen.grab_screen()
    old_shape = input.shape
    global input_shape
    input_shape = (1, old_shape[0], old_shape[1], 1)

    with tf.device('/cpu:0'):
        c = tf.placeholder(tf.float32, [None, 3])
        r = tf.placeholder(tf.float32, [None, 1])
        x = tf.placeholder(tf.float32, [None, input_shape[1], input_shape[2], input_shape[3]], name="input")
        y = model(x)
        o = tf.train.AdamOptimizer(optimizer_learning_rate)
        idk = y * c * r
        loss = tf.reduce_max(idk)
        min_me = 1 / tf.reduce_mean(loss)
        train = o.minimize(min_me)

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for t in range(0,3):
            time.sleep(1)
            print(3-t)
        for cycle in range(0, 100):
            resetGame()
            data = collect_data(sess, x, y)
            new_r = propagate_r_values(data[2])

            results = sess.run([loss, min_me, train], feed_dict={x: data[0], c: data[1], r: new_r})
            print(results[1])
            global data_collection_iterations
            if data_collection_iterations < 5000:
                data_collection_iterations += 100


if __name__ == '__main__':
    main()
    # input = screen.grab_screen()
    #
    # old_shape = input.shape
    # input_shape = (1, old_shape[0], old_shape[1], 1)
    # input.shape = input_shape
    # x = tf.placeholder(tf.float32, [None, old_shape[0], old_shape[1], 1], name="input")
    # y_out = model(x)
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     for t in range(0,3):
    #         time.sleep(1)
    #         print(3-t)
    #
    #     # input should be last 4ish frames.
    #     # Collect Data
    #     #   Put input into network
    #     #   Semi-randomly pick an output
    #     #   Record input, pick, r(None if it didn't change)
    #     # Propagate r values
    #     #   Start at end
    #     #   if r == None:
    #     #     r = last_r * 1/d
    #     #   else:
    #     #     reset
    #     # Train on Data
    #     #   prediction = model(input)
    #     #   loss = prediction * choice * r
    #
    #     # Repeat
    #
    #     for cycle in range(0,100):
    #         resetGame()
    #         frame_count = 0
    #         while input[0,256,32,0] > 1 or frame_count < 20:
    #             frame_count += 1
    #             input = screen.grab_screen()
    #             input.shape = input_shape
    #             time.sleep(0.001)
    #             result = sess.run(y_out, feed_dict={x: input})
    #             if result < -0.5:
    #                 directKey.ReleaseKey(d)
    #                 directKey.PressKey(a)
    #                 print('a')
    #             elif result > 0.5:
    #                 directKey.ReleaseKey(a)
    #                 directKey.PressKey(d)
    #                 print('d')
    #             else:
    #                 directKey.ReleaseKey(a)
    #                 directKey.ReleaseKey(d)
    #                 print(' ')
    #
    #         directKey.ReleaseKey(a)
    #         directKey.ReleaseKey(d)
    #         print(frame_count)
    #         print(1/frame_count)
    #         tf.train.AdamOptimizer(0.001).minimize(tf.constant(1/frame_count))
    #         time.sleep(0.5)

    # plt.show()

    # time.sleep(3)
    # print('Press')
    # while True:
    #     resetGame()
    #     time.sleep(2)


