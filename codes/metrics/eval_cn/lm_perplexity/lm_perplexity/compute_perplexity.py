import argparse
import json
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--perplexity_data_path', required=True)
    parser.add_argument('--utf8_conversion_scalar', default=None, type=float)
    parser.add_argument('--output_path', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    perplexity_data = torch.load(args.perplexity_data_path)
    all_aggregate_logprobs = np.concatenate(perplexity_data["all_logprobs"])
    sentences = perplexity_data["sentences"]
    aggregate_logprobs = perplexity_data["all_logprobs"]
    perplexity = [float(np.exp(-prob.mean())) for prob in aggregate_logprobs]
    total_ppl = float(np.exp(-all_aggregate_logprobs.mean()))
    ppl_dict = {}
    print(sentences)
    print(perplexity)
    for i, sent in enumerate(sentences):
        ppl_dict[sent] = perplexity[i]
    result = {
        # "perplexity": perplexity,
        "perplexity": ppl_dict,
        "total_ppl": total_ppl,
    }
    if args.utf8_conversion_scalar is not None:
        result["bpb"] = float(np.log2(perplexity) * args.utf8_conversion_scalar)
    if args.output_path:
        with open(args.output_path, "w") as f:
            f.write(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
