import tensorflow as tf
import tensorflow_probability as tfp

from softlearning.utils.tensorflow import cast_and_concat


tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors

def autoregressive_discrete_model(input_shape,
                                  hidden_layer_sizes,
                                  output_sizes,
                                  activation='relu',
                                  output_activation='linear',
                                  distribution_logits_activation='same'):
    """ 
    Returns two autoregressive models (sharing weights):
    1. A model that takes in the input features, but also the onehot samples for each dimension.
       Outputs the logits of each dimension.
    2. A model that takes in the input features only, samples each dimension from the logits to
       feed automatically into the next dimensions.
       Outputs the sampled onehot vectors for each dimension.
    """
    if distribution_logits_activation == 'same':
        distribution_logits_activation = output_activation

    inputs = tfk.Input(input_shape)

    inputs_given = []
    sampled_outputs = []
    logits_outputs = []

    for dim, output_size in enumerate(output_sizes):
        if dim == 0:
            X_given = inputs
            X_sampled = inputs
        else:
            # concatenate the previous dimensions to the input features
            concat_given = tfkl.Concatenate(axis=-1)([inputs] + inputs_given)
            concat_sampled = tfkl.Concatenate(axis=-1)([inputs] + sampled_outputs)
            X_given = concat_given
            X_sampled = concat_sampled
        
        # create MLP with shared layers
        for hidden_size in hidden_layer_sizes:
            hidden_layer = tfkl.Dense(hidden_size, activation=activation)
            X_given = hidden_layer(X_given)
            X_sampled = hidden_layer(X_sampled)

        output_layer = tfkl.Dense(output_size, activation='linear')
        output_given = output_layer(X_given)
        output_sampled = output_layer(X_sampled)

        # Get the logits output and create the input for training
        logits_output = tfkl.Activation(output_activation)(output_given)
        logits_outputs.append(logits_output)
        inputs_given.append(tfk.Input(output_size))
        
        # discrete sampling for policy
        distribution_logits = tfkl.Activation(distribution_logits_activation)(output_sampled)
        discrete_distribution = tfpl.OneHotCategorical(output_size)(distribution_logits)
        discrete_sample = tfkl.Lambda(lambda d: d.sample())(discrete_distribution)
        sampled_outputs.append(discrete_sample)

    model_given = tfk.Model([inputs] + inputs_given, logits_outputs)
    model_sampled = tfk.Model(inputs, sampled_outputs)

    return model_given, model_sampled

        


