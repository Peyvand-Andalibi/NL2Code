from __future__ import print_function
import theano


def inspect_inputs(i, node, fn):
    print(i, node, "input(s) value(s):", [input[0] for input in fn.inputs],
          end='')


def inspect_outputs(i, node, fn):
    print(" output(s) value(s):", [output[0] for output in fn.outputs])


def test_function(query_tokens, query_token_embed, query_embed, query_token_embed_mask, train_data):

    test_function_single = theano.function([query_tokens], query_embed, \
                                             mode=theano.compile.MonitorMode(
                                                 pre_func=inspect_inputs,
                                                 post_func=inspect_outputs))
    '''
    test_function_all = theano.function([query_tokens], query_token_embed, query_embed, query_token_embed_mask, \
                                                  mode=theano.compile.MonitorMode(
                                                      pre_func=inspect_inputs,
                                                      post_func=inspect_outputs))
    '''

    contents = test_function_single(train_data.examples[0].data[0])

    f = open("test_function_output.txt", "a")
    f.write("query tokens = %s" % str(train_data.examples[0].data[0]))
    f.write("\n\n")
    f.write("query embed = %s" % str(contents))
    for i in range(4):
        f.write("\n")
    f.close()

