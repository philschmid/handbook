import subprocess
import shlex
import logging

def execute_cli_script(script_path, args_dict, entrypoint="python"):
    """
    Executes a CLI script with given arguments, streaming logs to stdout.

    :param script_path: Path to the CLI script to execute.
    :param args_dict: Dictionary of arguments to pass to the script.
    :param entrypoint: The command to use to execute the script (e.g. "python").
    :raises: subprocess.CalledProcessError if the script execution fails.
    """
    # Construct the command with arguments
    command = [entrypoint, script_path]
    for key, value in args_dict.items():
        command.append(f"--{key}")
        command.append(str(value))
    
    # Ensure the command is safe for shell execution
    command = list(map(shlex.quote, command))
    logging.info(f"Executing: {' '.join(command)}")
    
    try:
        # Start the process
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
            # Stream output to stdout
            for line in process.stdout:
                print(line, end="")

            # Wait for the process to finish and get the exit code
            exit_code = process.wait()
            if exit_code != 0:
                # Read the remaining output
                stdout, stderr = process.communicate()
                # Raise an error with the output if the process fails
                raise subprocess.CalledProcessError(process.returncode, process.args, output=stdout, stderr=stdout)

    except subprocess.CalledProcessError as e:
        # Re-raise the error with the original context
        raise RuntimeError(f"An error occurred when executing the script: {e.stderr}") from e

