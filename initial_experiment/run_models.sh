echo "Starting script"

source ~/miniconda3/etc/profile.d/conda.sh

# echo "Running transformer_wide"
# conda run -n rl_project --no-capture-output python3 ~/CMPUT655-Project/initial_experiment/transformer_wide/experiment.py ~/CMPUT655-Project/initial_experiment/transformer_wide > ~/CMPUT655-Project/initial_experiment/transformer_wide/output.log 2> ~/CMPUT655-Project/initial_experiment/transformer_wide/error.log

# echo "Running transformer_length"
# conda run -n rl_project --no-capture-output python3 ~/CMPUT655-Project/initial_experiment/transformer_length/experiment.py ~/CMPUT655-Project/initial_experiment/transformer_length > ~/CMPUT655-Project/initial_experiment/transformer_length/output.log 2> ~/CMPUT655-Project/initial_experiment/transformer_length/error.log

# echo "Running lstm"
# conda run -n rl_project --no-capture-output python3 ~/CMPUT655-Project/initial_experiment/lstm/experiment.py ~/CMPUT655-Project/initial_experiment/lstm > ~/CMPUT655-Project/initial_experiment/lstm/output.log 2> ~/CMPUT655-Project/initial_experiment/lstm/error.log

# echo "Running transformer_wide_2"
# conda run -n rl_project --no-capture-output python3 ~/CMPUT655-Project/initial_experiment/transformer_wide_2/experiment.py ~/CMPUT655-Project/initial_experiment/transformer_wide_2 > ~/CMPUT655-Project/initial_experiment/transformer_wide_2/output.log 2> ~/CMPUT655-Project/initial_experiment/transformer_wide_2/error.log

# echo "Running lstm_2"
# conda run -n rl_project --no-capture-output python3 ~/CMPUT655-Project/initial_experiment/lstm_2/experiment.py ~/CMPUT655-Project/initial_experiment/lstm_2 > ~/CMPUT655-Project/initial_experiment/lstm_2/output.log 2> ~/CMPUT655-Project/initial_experiment/lstm_2/error.log

# echo "Running gru"
# conda run -n rl_project --no-capture-output python3 ~/CMPUT655-Project/initial_experiment/gru/experiment.py ~/CMPUT655-Project/initial_experiment/gru > ~/CMPUT655-Project/initial_experiment/gru/output.log 2> ~/CMPUT655-Project/initial_experiment/gru/error.log

echo "Running gru_all_env"
conda run -n rl_project --no-capture-output python3 ~/CMPUT655-Project/initial_experiment/gru_all_env/experiment.py ~/CMPUT655-Project/initial_experiment/gru_all_env 0 > ~/CMPUT655-Project/initial_experiment/gru_all_env/output.log 2> ~/CMPUT655-Project/initial_experiment/gru_all_env/error.log

echo "Running lstm_all_env"
conda run -n rl_project --no-capture-output python3 ~/CMPUT655-Project/initial_experiment/lstm_all_env/experiment.py ~/CMPUT655-Project/initial_experiment/lstm_all_env 1 > ~/CMPUT655-Project/initial_experiment/lstm_all_env/output.log 2> ~/CMPUT655-Project/initial_experiment/lstm_all_env/error.log

echo "Running gtrxl_all_env"
conda run -n rl_project --no-capture-output python3 ~/CMPUT655-Project/initial_experiment/gtrxl_all_env/experiment.py ~/CMPUT655-Project/initial_experiment/gtrxl_all_env 2 > ~/CMPUT655-Project/initial_experiment/gtrxl_all_env/output.log 2> ~/CMPUT655-Project/initial_experiment/gtrxl_all_env/error.log

echo "Running deep_linear_attention_all_env"
conda run -n rl_project --no-capture-output python3 ~/CMPUT655-Project/initial_experiment/deep_linear_attention_all_env/experiment.py ~/CMPUT655-Project/initial_experiment/deep_linear_attention_all_env 3 > ~/CMPUT655-Project/initial_experiment/deep_linear_attention_all_env/output.log 2> ~/CMPUT655-Project/initial_experiment/deep_linear_attention_all_env/error.log

echo "Ending script"
