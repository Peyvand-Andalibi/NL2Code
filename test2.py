from __future__ import print_function
import theano
import theano.tensor as T
import model
from nn.utils.theano_utils import *
import numpy as np

def inspect_inputs(i, node, fn):
    print(i, node, "input(s) value(s):", [input[0] for input in fn.inputs],
          end='')


def inspect_outputs(i, node, fn):
    print(" output(s) value(s):", [output[0] for output in fn.outputs])


def write_to_file(inputs, outpus):
    f = open("test_function_output.txt", "a")
    f.write("inputs = %s" % str(inputs))
    f.write("\n\n")
    f.write("outputs = %s" % str(outpus))
    for i in range(4):
        f.write("\n")
    f.close()


def test_transposed(input, output, data):

    function = theano.function([input], output, \
                                             mode=theano.compile.MonitorMode(
                                                 pre_func=inspect_inputs,
                                                 post_func=inspect_outputs))
    '''
    #tensors
    query_tokens = ndim_itensor(2, 'query_tokens')
    query_token_embed, _ = model.Model(train_data).query_embedding(query_tokens, mask_zero=True)
    embed_function = theano.function([query_tokens],[query_token_embed])

    #arrays
    embedded_query = embed_function(train_data.examples[0].data[0])
    embedded_transposed_query = np.transpose(embedded_query,(1, 0, 2))
    '''
    transposed_data = np.transpose(data,(1, 0, 2))
    write_to_file(transposed_data, function(transposed_data))


def test_shape_transposed(input, output, data):
    function = theano.function([input], output)

    '''
    # tensors
    query_tokens = ndim_itensor(2, 'query_tokens')
    query_token_embed, _ = model.Model(train_data).query_embedding(query_tokens, mask_zero=True)
    embed_function = theano.function([query_tokens], [query_token_embed])

    # arrays
    embedded_query = embed_function(train_data.examples[0].data[0])
    embedded_transposed_query = np.transpose(embedded_query,(1, 0, 2))
    '''
    transposed_data = np.transpose(data, (1, 0, 2))
    print(np.shape(function(transposed_data)))

def test_general(input, output, input_value):
    function = theano.function([input], output, \
                               mode=theano.compile.MonitorMode(
                                   pre_func=inspect_inputs,
                                   post_func=inspect_outputs))

    write_to_file(input_value, function(input_value))


