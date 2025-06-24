import argparse
import torch
from tqdm import auto as tqdm_lib
import sys
sys.path.append("./")
import json
import lm_dataformat
import lm_perplexity.models as models
import lm_perplexity.utils as utils
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config_path', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--max_docs', type=int, default=None)
    parser.add_argument('--doc_indices_path', type=str, default=None)
    parser.add_argument('--utf8_conversion_scalar', default=None, type=float)
    parser.add_argument('--output_path', default=None)
    return parser.parse_args()

def read_json_line(path):
    data = []
    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line.strip()))
    return data

def compute_perplexity_data(model, data_path, indices=None):
    # For expedience, we're going to assume everything fits in memory for now
    # Also for expedience we're just going to save lists of arrays
    overall_output = {
        "sentences": [],
        "all_logprobs": [],
        "all_positions": [],
        "aggregate_length": 0,
        "aggregate_utf8_length": 0.
    }

    if "local" in data_path or "all" in data_path or "global" in data_path:
        rawdata = read_json_line(data_path)
        data = []
        for jterm in rawdata:
            data.append(jterm["newcap_generated"])
    else:
        with open(data_path, "r") as f:
            data = json.load(f)

    for i, doc in enumerate(tqdm_lib.tqdm(data)):
        if indices is not None and i not in indices:
            continue
        output = model.get_perplexity_data(doc)
        if not output:
            continue
        overall_output["sentences"].append(doc)
        overall_output["all_logprobs"].append(output["logprobs"])
        overall_output["all_positions"].append(output["positions"])
        overall_output["aggregate_length"] += output["length"]
        overall_output["aggregate_utf8_length"] += output["utf8_length"]

    return overall_output


def main():
    args = parse_args()
    model = models.create_model(args.model_config_path)
    if args.doc_indices_path:
        assert args.max_docs is None
        indices = set(utils.read_json(args.doc_indices_path))
    elif args.max_docs:
        assert args.doc_indices_path is None
        indices = set(range(args.max_docs))
    else:
        indices = None
    perplexity_data = compute_perplexity_data(
        model=model,
        data_path=args.data_path,
        indices=indices,
    )

    all_aggregate_logprobs = np.concatenate(perplexity_data["all_logprobs"])
    sentences = perplexity_data["sentences"]
    aggregate_logprobs = perplexity_data["all_logprobs"]
    perplexity = [float(np.exp(-prob.mean())) for prob in aggregate_logprobs]
    total_ppl = float(np.exp(-all_aggregate_logprobs.mean()))
    ppl_dict = {}
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

    print("PPL {:.1f}".format(result["total_ppl"]))
        

if __name__ == "__main__":
    main()
