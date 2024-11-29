import argparse
import subprocess
import os
import logging

def setup_logging(base_path):
    """Set up logging to save logs in the log/ directory under base_path."""
    log_dir = os.path.join(base_path, "log")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "pipeline_execution.log")
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info("Logging initialized.")

import subprocess
import logging

def run_command(command, step_name):
    """Helper function to run shell commands, log results, and show output in real-time."""
    try:
        logging.info(f"Starting step: {step_name}")
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        # Stream the output
        for line in process.stdout:
            print(line, end='')  # Print to standard output
            logging.info(line.strip())  # Log to file

        for line in process.stderr:
            print(line, end='')  # Print to standard output
            logging.error(line.strip())  # Log to file

        # Wait for the process to finish and check for errors
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
        
        logging.info(f"Step {step_name} completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Step {step_name} failed with return code {e.returncode}.")
        print(f"An error occurred during step: {step_name}. Check the log for details.")
        exit(1)

def main(args):
    # Determine the base path dynamically
    base_path = os.getcwd()

    # Infer paths if not provided
    base_output_path = args.output_path_base or os.path.join(base_path, "output", args.model_name)
    dataset_dir = args.dataset_dir or os.path.join(base_output_path, "dataset/multi_conversation_model/multi_news")
    grad_dir = args.grad_dir or os.path.join(base_output_path, "profile")
    importance_tensor_dir = args.importance_tensor_dir or os.path.join(base_output_path, "profile/")
    moa_config_dir = args.moa_config_dir or os.path.join(base_output_path, "optimize")
    result_dir = args.result_dir or os.path.join(base_output_path, "validate")
    mha_model_path = args.mha_model_path or os.path.join(base_output_path, "mha_model", args.model_name)

    setup_logging(base_output_path)

    # File to track completed steps
    completed_steps_file = os.path.join(base_output_path, "pipeline_completed_steps.txt")
    completed_steps_dir = os.path.dirname(completed_steps_file)
    os.makedirs(completed_steps_dir, exist_ok=True)  # Ensure the directory exists
    
    if not os.path.exists(completed_steps_file):
        open(completed_steps_file, 'w').close()

    # Helper to check if a step was completed
    def is_step_completed(step_name):
        with open(completed_steps_file, 'r') as f:
            return step_name in f.read()

    # Helper to mark a step as completed
    def mark_step_completed(step_name):
        with open(completed_steps_file, 'a') as f:
            f.write(f"{step_name}\n")

    # Step 1: Generate Calibration Dataset
    step_name = "Generate Calibration Dataset"
    if not is_step_completed(step_name):
        print(f"Executing: {step_name}")
        calibration_command = f"python scripts/pipeline/generate_calibration_dataset.py " \
                              f"--model_path {args.model_path} " \
                              f"--model_name {args.model_name} " \
                              f"--output_path_base {os.path.join(base_output_path, 'dataset')}"
        print(calibration_command)
        run_command(calibration_command, step_name)
        mark_step_completed(step_name)
    else:
        print(f"Skipping {step_name}, already completed.")

    # Optional Step: Convert to MHA
    if args.is_gqa:
        step_name = "Convert to MHA"
        if not is_step_completed(step_name):
            print(f"Executing: {step_name}")
            convert_command = f"python scripts/helper/gqa_to_mha.py " \
                            f"--model_path {args.model_path} " \
                            f"--output_path {mha_model_path}"
            run_command(convert_command, step_name)
            mark_step_completed(step_name)
            # Update the model_path to the converted model
            args.model_path = mha_model_path

    # Step 2: Profile
    step_name = "Profile Model"
    if not is_step_completed(step_name):
        print(f"Executing: {step_name}")
        for max_length in args.elastic_lengths.split(' '):
            max_length = int(max_length.strip())
            profile_command = f"python scripts/pipeline/pipeline_profile.py " \
                             f"--model_name {args.model_path} " \
                             f"--max_length {max_length} " \
                             f"--response_mask " \
                             f"--dataset_dir {dataset_dir} " \
                             f"--grad_dir {os.path.join(grad_dir, f'profile_{int(max_length/1024)}k')}"
            print(profile_command)
            run_command(profile_command, f"{step_name} for max_length={max_length}")
        mark_step_completed(step_name)
    else:
        print(f"Skipping {step_name}, already completed.")

    # Step 3: Optimize
    step_name = "Optimize Model"
    if not is_step_completed(step_name):
        print(f"Executing: {step_name}")
        optimize_command = f"python scripts/pipeline/elastic_generate.py " \
                           f"--output_dir {base_output_path} " \
                           f"--elastic_length {args.elastic_lengths} " \
                           f"--extend_length {args.extend_length} " \
                           f"--density_bounds {args.density_bounds} " \
                           f"--importance_tensor_dir {importance_tensor_dir} " \
                           f"--output_length {args.output_lengths}"
        print(optimize_command)
        run_command(optimize_command, step_name)
        mark_step_completed(step_name)
    else:
        print(f"Skipping {step_name}, already completed.")

    # Step 4: Validate
    step_name = "Validate Model"
    if not is_step_completed(step_name):
        print(f"Executing: {step_name}")
        # get moa config num by counting the number of json files in the moa_config_dir
        moa_config_num = len([name for name in os.listdir(moa_config_dir) if name.endswith(".json")])
        validate_command = f"scripts/pipeline/validate.sh {moa_config_dir} " \
                           f"{moa_config_num} " \
                           f"{result_dir} " \
                           f"{args.model_path}"
        print(validate_command)
        run_command(validate_command, step_name)
        mark_step_completed(step_name)
    else:
        print(f"Skipping {step_name}, already completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the automatic search pipeline for MoA model compression.")
    # required args
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model on local disk or huggingface, for example lmsys/vicuna-7b-v1.5-16k")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model for saving directories, for example lmsys--vicuna-7b-v1.5-16k")

    # settings
    parser.add_argument("--is_gqa", action="store_true", help="Convert the model to MHA format if True")
    parser.add_argument("--elastic_lengths", type=str, default="2048 4096 8192", help="Elastic lengths used for profiling")
    parser.add_argument("--extend_length", type=str, default="16384", help="Maximum length for the compression plan")
    parser.add_argument("--density_bounds", type=str, default="0.5 0.5 0.5 0.5", help="Density bounds")
    parser.add_argument("--output_lengths", type=str, default="12288 16384", help="Output reference lengths")
    
    # paths that can also be inferred if not provided
    parser.add_argument("--output_path_base", type=str, help="Base output path for the dataset")
    parser.add_argument("--moa_config_dir", type=str, help="Directory for MoA config")
    parser.add_argument("--result_dir", type=str, help="Directory for results")
    parser.add_argument("--dataset_dir", type=str, help="Directory for the dataset")
    parser.add_argument("--grad_dir", type=str, help="Directory for gradient files")
    parser.add_argument("--importance_tensor_dir", type=str, help="Directory for importance tensors")
    parser.add_argument("--mha_model_path", type=str, help="Path to the MHA model")
    
    args = parser.parse_args()
    main(args)