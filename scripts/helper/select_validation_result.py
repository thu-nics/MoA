import pandas as pd
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)

    args = parser.parse_args()

    input_dir = args.input_dir

    min = 100
    min_name = ""

    for file_name in os.listdir(input_dir):
        if file_name.endswith('csv'):
            # use pandas to open it
            df = pd.read_csv(os.path.join(input_dir, file_name))
            mean_ce = df['mean'].item()
            if mean_ce < min:
                min = mean_ce
                min_name = file_name

    print(f"{min_name} has the lowest loss")