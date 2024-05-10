import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence, cast
from alignment.utils import execute_cli_script
from trl import TrlParser

# get current abolute path of the file but without the file
absoulte_path = os.path.dirname(os.path.realpath(__file__))

# python /fsx/philipp/alignment-handbook/scripts/iterative_dpo/main.py --model_name_or_path alignment-handbook/zephyr-7b-sft-full --dataset_path philschmid/dolly-15k-oai-style --output_dir /fsx/philipp/alignment-handbook --batch_size 8

@dataclass
class CandidateArguments:
    model_name_or_path: str
    dataset_path: str
    output_dir: str
    num_samples: int = 5
    batch_size: int = 1
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_k: int = -1
    top_p: float = 1.0
    
    
def main():
  parser = TrlParser((CandidateArguments))
  script_args = parser.parse_args_and_config()[0]
  script_args = cast(CandidateArguments, script_args)

  # Generate Candidates
  script_path = os.path.join(absoulte_path,"run_generate_candidates.py")
  execute_cli_script(
    script_path,
    asdict(script_args)
  )


if __name__ == "__main__":
    main()