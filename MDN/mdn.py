"""Multivariate Mixture Density Network"""
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utils import linear


def gausspdf(x, mean, sigma):
    """Returns the multivariate gaussian pdf.
    Parameters
    ----------
    x: Tensor.
        (n_batch, 1, 1, n_channels) tensor.

    mean: Tensor.
        (n_batch, n_gauss, 1, n_channels) tensor.

    sigma: Tensor.
        (n_batch, n_gauss, n_channels, n_channels) varcov tensor.

    Returns
    -------
    gausspdf: Tensor.
        Return the multivariate gaussian pdf for n batches.
    """
    den = (2 * np.pi) ** x.get_shape().as_list()[1]
    den = tf.sqrt(tf.matrix_determinant(den * sigma))
    num = - 0.5 * tf.matmul((x - mean), tf.matrix_inverse(sigma))
    num = tf.exp(tf.matmul(num, tf.matrix_transpose(x - mean)))
    num = tf.squeeze(num, [2, 3])
    pdf = num / den

    return pdf


def create_model(n_neurons, activation, lr, img_shape, n_gaussians, varcov):
    """Creates a model that predicts RGB values of an image.

    Parameters
    ----------
    n_neurons: list.
        List containing the number of neurons per layer.

    activation: tf.nn.activation.
        Tensorflow activation for each dense layer.

    lr: float.
        Learning rate.

    img_shape: tuple.
        Tuple defining the dimensions of the image.

    n_gaussians: int.
        Number of normal distributions per image channel.

    varcov: string.
        String specifying the type of varcov for the normal distributions.
        "iso" for an isotropic varcov.
        "diag" for a diagonal varcov.
        "norm" for a varcov with non-zero covariances.

    Returns
    -------
    dictionary: dict
        The dictionary contains the following elements of the graph.
        - X
        - Y
        - loss
        - train_op
        - init_op
        - means
        - sigmas
        - weights
        - pdf
        - global_step
    """
    n_channels = img_shape[-1]

    X = tf.placeholder(tf.float32, shape=[None, 2], name="X")
    Y = tf.placeholder(tf.float32, shape=[None, 3], name="Y")

    # Creating the global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Current input
    h = X

    # Creating all the dense layers
    for i, neurons in enumerate(n_neurons):
        with tf.variable_scope("dense{}".format(i)):
            h, W = linear(h, neurons, activation)
            tf.summary.histogram(name="output_layer{}".format(i), values=h)

    # Gaussian mixture.
    # Means.
    with tf.variable_scope("means"):
        means, W = linear(h, n_channels * n_gaussians, activation)
        means = tf.reshape(means,
                           [-1, n_gaussians, 1, n_channels],
                           name="means_reshape")

    # Isotropic var-cov.
    with tf.variable_scope("sigmas"):
        if varcov == "iso":
            # Shape [n_batch, n_gaussians].
            sigmas, W = linear(h,
                               n_gaussians,
                               activation)
            sigmas = tf.maximum(sigmas, 1e-10)

            # Shape [n_batch, n_gaussians, 1].
            sigmas = tf.expand_dims(sigmas, axis=[-1])

            # Shape [n_batch, n_gaussians, n_channels].
            sigmas = tf.tile(sigmas, [1, 1, n_channels])

            # Shape [n_batch, n_gaussians, n_channels, n_channels].
            sigmas = tf.matrix_diag(sigmas)

        elif varcov == "diag":
            # Shape [n_batch, n_channels * n_gaussians]
            sigmas, W = linear(h,
                               n_gaussians * n_channels,
                               activation)
            sigmas = tf.maximum(sigmas, 1e-10)

            # Shape [n_batch, n_gaussians, n_channels]
            sigmas = tf.reshape(sigmas, [-1, n_gaussians, n_channels])

            # Shape [n_batch, n_gaussians, n_channels, n_channels]
            sigmas = tf.matrix_diag(sigmas)

        else:
            # Non-zero covariance matrix
            # Shape [n_batch, n_gaussians * n_channels * n_channels]
            sigmas, W = linear(h,
                               n_gaussians * n_channels ** 2,
                               activation)
            sigmas = tf.maximum(sigmas, 1e-10)

            # Shape [n_batch, n_gaussians , n_channels , n_channels]
            sigmas = tf.reshape(sigmas,
                                [-1, n_gaussians, n_channels, n_channels])

            # Note that the above tensor does not
            # ensure that the varcov matrix
            # is symmetric. So we have to do an
            # inefficient trick to ensure symmetry.
            # See https://stackoverflow.com/questions/36697736/how-to-force-tensorflow-tensors-to-be-symmetric
            # Also, theres no way to ensure that the resulting
            # varcov is a nonsingular matrix.
            # So theres no guarantee that the tf.matrix_inverse
            # operation in the gausspdf function returns
            # anything but garbage.
            sigmas = 0.5 * (sigmas + tf.matrix_transpose(sigmas))

    # Calculating the weight for each gaussian.
    with tf.variable_scope("weights"):
        weights, W = linear(h,
                            n_gaussians,
                            tf.nn.softmax)

    # Reshape Y so that it matches the shape of mu.
    Y_4d = tf.reshape(Y, shape=[-1, 1, 1, n_channels], name="Y_4d")

    # Calculating the mixture prob.
    with tf.variable_scope("mixture"):
        pdf = gausspdf(Y_4d, means, sigmas)
        mixture = tf.multiply(pdf, weights)
        mixture = tf.reduce_sum(mixture, axis=[1])

    # Defining loss function, train_op and init_op
    # Negative loglikelihood.
    loss = - tf.reduce_mean(tf.log(tf.maximum(mixture, 1e-10)))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss,
                                                   global_step=global_step)
    init_op = tf.global_variables_initializer()

    # Summaries.
    tf.summary.scalar(name="negloglike", tensor=loss)
    tf.summary.histogram(name="sigmas", values=sigmas)
    tf.summary.histogram(name="means", values=means)

    return {"X": X,
            "Y": Y,
            "loss": loss,
            "train_op": train_op,
            "init_op": init_op,
            "means": means,
            "sigmas": sigmas,
            "weights": weights,
            "pdf": pdf,
            "step": global_step}


def train_model(n_neurons,
                activation,
                lr,
                xs,
                ys,
                n_gaussians,
                varcov,
                batch_size,
                epochs,
                img_shape,
                dir_n,
                rest_path=None):
    """ Trains the model.

    Parameters
    ----------
    n_neurons: list.
        List containing the number of neurons per layer.

    activation: tf.nn.activation.
        Tensorflow activation for each dense layer.

    lr: float.
        Learning rate.

    xs: list.
        List containing the (X,Y) points in a plane.

    ys: list.
        List containing the corresponding RGB values for every
        element in xs.

    n_gaussians: int.
        Number of multivariate gaussians in the model.

    varcov: string.
        String specifying the type of varcov for the normal distributions.
        "iso" for an isotropic varcov.
        "diag" for a diagonal varcov.
        "norm" for a varcov with non-zero covariances.

    batch_size: int.
        Batch size for training.

    epochs: int.
        Number of epochs for training.

    img_shape: tuple.
        Tuple defining the shape of the image.

    dir_n: string.
        Model name used to save files.

    rest_path: string.
        Path for restoring the model.
    """
    # Defining the name of the model.
    model_name = "mdn" + "_varcov:" + varcov + "_ng:" \
                 + str(n_gaussians) + "_bz:" + str(batch_size) \
                 + "_lr:" + str(lr) + "_imgs:" + str(xs.shape[0] / (250 * 187))

    activation = tf.nn.relu

    g = tf.Graph()
    with tf.Session(graph=g) as sess:

        # Creating the model.
        model = create_model(n_neurons,
                             activation,
                             lr,
                             img_shape,
                             n_gaussians,
                             varcov)

        # Creating the saver operation.
        saver = tf.train.Saver()

        # Initialize variables or restore them.
        if rest_path is None:
            sess.run(model["init_op"])
        else:
            saver.restore(sess, rest_path)
            print("Model restored.")

        # Merging the summaries.
        sums = tf.summary.merge_all()

        # Defining the writer operation.
        train_writer = tf.summary.FileWriter(logdir="logdir/" + model_name,
                                             graph=g)

        # Training.
        for epoch in range(epochs):
            idxs = np.random.permutation(range(len(xs)))
            n_batches = len(idxs) // batch_size

            for batch_i in range(n_batches):
                idxs_i = idxs[batch_i * batch_size: (batch_i + 1) * batch_size]
                cost = sess.run([model["loss"], model["train_op"]],
                                feed_dict={model["X"]: xs[idxs_i],
                                           model["Y"]: ys[idxs_i]})[0]

                step = model['step'].eval(session=sess)
                print("Epoch:", epoch, "Global step:", step, "Cost:", cost)

            # Saving the model and reconstructing the img.
            if (epoch + 1) % 500 == 0:
                # Getting the summary and recon.
                summary = sess.run([sums], feed_dict={model["X"]: xs[idxs_i],
                                                      model["Y"]: ys[idxs_i]})[0]

                # Adding a summary data point.
                train_writer.add_summary(summary, epoch)

            if (epoch + 1) % 100 == 0:
                means, weights = sess.run([model["means"], model["weights"]],
                                          feed_dict={model["X"]: xs[: 250 * 187]})

                # Making the reconstruction.
                means = means[:, :, 0, :]  # Deleting the axis with 1 dim.

                w_argmax = weights.argmax(axis=1)

                means = np.array([means[obv, idx, :]
                                  for obv, idx in enumerate(w_argmax)])

                means = np.clip(means, 0, 1)
                means = means.reshape([250, 187, 3])

                plt.imsave("recons/{}/recon_{:08d}.png".format(dir_n, epoch),
                           means)

                # Saving the model.
                save_path = saver.save(
                    sess,
                    "models/{}/basic.ckpt".format(dir_n),
                    global_step=epoch)

                print("Model saved in file: {}".format(save_path))


def main():
    """Do a grid search over the model."""
    imgs = [plt.imread("g1.png"), 
            plt.imread("g2.png")]
    # Getting xs and ys.
    xs = []
    ys = []

    for img in imgs:
        for row_i in range(img.shape[0]):
            for col_i in range(img.shape[1]):
                xs.append([row_i, col_i])
                ys.append(img[row_i, col_i])

    # Transforming X and Y to arrays.
    xs = np.array(xs)
    ys = np.array(ys)

    # Normalizing input data.
    xs = (xs - np.mean(xs)) / np.std(xs)
    ys = ys  # / 255.0

    # Static parameters.
    n_neurons = [256] * 6
    activation = "relu"
    epochs = 10000

    # Parameters for the grid search.
    n_gaussians = [14, 100]
    varcovs = ["diag", "iso"]
    lrs = [0.0001]
    batch_sizes = [250 * 187]

    # Counter for the model.
    n = 9

    for n_gaussian in n_gaussians:
        for varcov in varcovs:
            for batch_size in batch_sizes:
                for lr in lrs:
                    # Creating a directory for each model.
                    dir_n = "mmds{}".format(n)
                    os.makedirs("models/{}".format(dir_n))
                    os.makedirs("recons/{}".format(dir_n))
                    print("Directory created for:", dir_n)

                    # Saving a 'dictionary' containing
                    # each dir_n and its parameters.
                    model_params = "varcov:" + varcov + "_ng:" \
                                   + str(n_gaussian) + "_bz:" + str(batch_size) \
                                   + "_imgs:" + str(len(imgs)) + "_lr:" + str(lr)

                    with open("mmdn_des.txt", "a") as file:
                        file.write(dir_n + " " + model_params + "\n")

                    print(dir_n, "added to models_des.txt")

                    # Training the model.
                    train_model(n_neurons,
                                activation,
                                lr,
                                xs,
                                ys,
                                n_gaussian,
                                varcov,
                                batch_size,
                                epochs,
                                img.shape,
                                dir_n)

                    # Updating counter.
                    n += 1


if __name__ == '__main__':
    main()
