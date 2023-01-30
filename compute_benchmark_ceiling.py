from neural_nlp import benchmark_pool
# create an argument parser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--bechmark', type=str, default='LangLocECoGv2-encoding')

if __name__ == "__main__":
    args = parser.parse_args()
    benchmark_name = args.bechmark
    benchmark = benchmark_pool[benchmark_name]
    benchmark.ceiling





