import torch
# from transformers import AutoTokenizer, AutoModelWithLMHead

# tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
# model = AutoModelWithLMHead.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

from transformers import BertTokenizer, GPT2LMHeadModel
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

def calculate_ppl(model, input_text):
    input_ids = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs[0]
    ppl = torch.exp(loss)
    return ppl

input_text = "你好，这是一个测试句子。"
ppl_score = calculate_ppl(model, input_text)
print("Perplexity score:", float(ppl_score))
