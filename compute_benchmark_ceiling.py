from neural_nlp import benchmark_pool
# create an argument parser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str, default='LangLocECoGv2-encoding')

if __name__ == "__main__":
    args = parser.parse_args()
    benchmark_name = args.benchmark
    benchmark = benchmark_pool[benchmark_name]
    benchmark.ceiling_estimate





