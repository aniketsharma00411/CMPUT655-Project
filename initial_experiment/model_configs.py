# Transformer wide -  params 693957
model = {"use_attention": True,
         "max_seq_len": 30,
         # "custom_model_config":{
         'attention_num_transformer_units': 2,
         'attention_dim': 64,
         'attention_num_heads': 20,
         # 'memory_tau':50
         # }
         }

# ---------------------------------------------------------------------

# Transformer Length / Transformer Narror - params 694373
model = {"use_attention": True,
         "max_seq_len": 30,
         # "custom_model_config":{
         'attention_num_transformer_units': 7,
         'attention_dim': 64,
         'attention_num_heads': 2,
         # 'memory_tau':50
         # }
         }

# ---------------------------------------------------------------------

# LSTM Default - params 685317
model = {"use_lstm": True,
         "max_seq_len": 30,
         'lstm_cell_size': 256
         }

# ---------------------------------------------------------------------

# Transformer wide 2
model = {"use_attention": True,
         "max_seq_len": 30,
         # "custom_model_config":{
         'attention_num_transformer_units': 1,
         'attention_dim': 32,
         'attention_num_heads': 2,
         'fcnet_hiddens': [32],
         # 'memory_tau':50
         # }
         }

# ---------------------------------------------------------------------

# LSTM 2
model = {"use_lstm": True,
         "max_seq_len": 30,
         'lstm_cell_size': 64,
         'fcnet_hiddens': [32],
         }

# ---------------------------------------------------------------------
