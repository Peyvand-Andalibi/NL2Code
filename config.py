mode = 'django'

source_vocab_size = 2490 # 2492 # 5980
target_vocab_size = 2101 # 2110 # 4830 #
rule_num = 222 # 228
node_num = 96

node_embed_dim = 256
embed_dim = 128
rule_embed_dim = 256
query_dim = 256
lstm_state_dim = 128
decoder_att_hidden_dim = 50
pointer_net_hidden_dim = 50

max_query_length = 70
max_example_action_num = 100

decoder_dropout = 0.2
word_dropout = 0

# encoder
encoder_lstm = 'lstm'
encoder_kernel_size = 3

# decoder
parent_hidden_state_feeding = True
parent_rule_feeding = True
node_type_feeding = True
tree_attention = True

# training
train_patience = 10
max_epoch = 50
batch_size = 10
valid_per_minibatch = 4000
save_per_minibatch = 4000

# decoding
beam_size = 15
decode_max_time_step = 100

config_info = None
operation = ''