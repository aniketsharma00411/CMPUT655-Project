echo "Starting script"

source ~/miniconda3/etc/profile.d/conda.sh

echo "Running transformer_wide"
conda run -n rl_project --no-capture-output python3 ~/CMPUT655-Project/initial_experiment/transformer_wide/experiment.py ~/CMPUT655-Project/initial_experiment/transformer_wide > ~/CMPUT655-Project/initial_experiment/transformer_wide/output.log 2> ~/CMPUT655-Project/initial_experiment/transformer_wide/error.log

echo "Running transformer_length"
conda run -n rl_project --no-capture-output python3 ~/CMPUT655-Project/initial_experiment/transformer_length/experiment.py ~/CMPUT655-Project/initial_experiment/transformer_length > ~/CMPUT655-Project/initial_experiment/transformer_length/output.log 2> ~/CMPUT655-Project/initial_experiment/transformer_length/error.log

echo "Running lstm"
conda run -n rl_project --no-capture-output python3 ~/CMPUT655-Project/initial_experiment/lstm/experiment.py ~/CMPUT655-Project/initial_experiment/lstm > ~/CMPUT655-Project/initial_experiment/lstm/output.log 2> ~/CMPUT655-Project/initial_experiment/lstm/error.log

echo "Ending script"
