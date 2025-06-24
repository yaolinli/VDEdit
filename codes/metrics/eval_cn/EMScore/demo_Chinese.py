'''
conda env: base
use CN-CLIP transformers feature
'''
# from scorer import EMScorer
# from emscore_ecommerce import EMScorer
from emscore_Chinese import EMScorer
from emscore_Chinese.utils import get_idf_dict, compute_correlation_uniquehuman
import pdb

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
      
train_corpus_path = "EMMAD-EVAL/EMMAD_train_data.txt"
caps = []
with open(train_corpus_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        terms = line.strip().split("\t")
        cap = terms[-1]
        caps.append(cap)
        
vatex_train_corpus_list = caps
emscore_idf_dict = get_idf_dict(vatex_train_corpus_list, tokenizer, nthreads=4, language='cn')
# max token_id are eos token id
# set idf of eos token are mean idf value
emscore_idf_dict[max(list(emscore_idf_dict.keys()))] = sum(list(emscore_idf_dict.values()))/len(list(emscore_idf_dict.values()))


if __name__ == '__main__':
    # you video's path list
    metric = EMScorer()
    
    vids = [
            ['emscore_ecommerce/example/214491433575.mp4'],
            ['emscore_ecommerce/example/ecommerce.mp4'], 
            ]
    raw_cands = [[
        '这款UN##NY卸妆水在我心目中已经完美取代贝德玛卸妆水了，它不含酒精、色素、矿物油等有害成分，同时还有薰衣草、虎杖根等天然成分，卸妆过程更是一场护肤过程。成分温和不刺激，不仅卸妆力非常优秀，卸完之后肌肤滋润，敏感肌和准妈妈都可以放心使用。',
        '这款UNNY卸妆水在我心目中已经完美取代贝德玛卸妆水了，它不含酒精、色素、矿物油等有害成分，同时还有薰衣草、虎杖根、迷迭香等天然成分，卸妆过程更是一场护肤过程。成分温和不刺激，不仅卸妆力度也是非常优秀，卸完之后肌肤滋润不紧绷，敏感肌和准妈妈都可以放心使用。',
        '这款UN##NY卸妆水在我心目中已经完美取代贝德玛卸妆油了，它不含酒精、色素、矿物油等有害成分，同时还有薰衣草、虎杖根、迷迭香等天然成分。成分温和不刺激，不仅卸妆力非常优秀，卸完之后肌肤滋润不紧绷，敏感肌和准妈妈都可以放心使用。'
        ],
        [
        '这款人气的宽松针织运动裤，高档的面料，柔软舒适，耐穿耐磨，立体裁剪，高端缝制，精工细作，恰到好处的尊贵，阳光又青春，深受学生党的喜爱。只要稍微简单搭配，看看这款好搭配又不会沉闷，让你尽显型男气质哦。',
        '这款人气的宽松针织运动裤，高档的面料，柔软舒适，耐穿耐磨，立体裁剪，高端缝制，精工细作，恰到好处的尊贵，阳光又青春，深受学生党的喜爱。',
        '只要稍微简单搭配，看看这款好搭配又不会沉闷，让你尽显型男气质哦。', 
        '这款人气的宽松针织裙子，高档的面料，柔软舒适，耐穿耐磨，立体裁剪，高端缝制，精工细作，恰到好处的尊贵，阳光又青春，深受女学生的喜爱。',
        '这款人气的宽松针织运动裤，阳光又青春，精工细作，粉色配色，深受学生党的喜爱。', 
        '这款人气的宽松针织运动裤，阳光又青春，精工细作，黑白配色，深受学生党的喜爱。', 
        '淡水珍珠锁骨链，颈间的一抹小清新，多粒大小不一珍珠的串联，将你的气质衬托出来，爱美小仙女都在买！',
        '这款人气的宽松针织运动裤，阳光又青春，精工细作，黑白配色，到脚踝的长度很显瘦，深受学生党的喜爱。', 
        '这款人气的宽松针织运动裤，阳光又青春，精工细作，黑白配色，长度短，深受学生党的喜爱。', 
        '这款人气的宽松针织运动裤，阳光又青春，精工细作，黑白配色，长度长，深受学生党的喜爱。 ',
        ],
    ]
    for cand in raw_cands:
        cands = [cand]
        refs = []
        refs = []
        # results = metric.score(cands=cands, refs=refs, vids=vids, idf=False)
        results = metric.score(cands=cands, refs=refs, vids=vids, idf=emscore_idf_dict)
        print("****************")
        print("[{:.4f}] {}".format(results['EMScore(X,V)']['full_F'].item(), cands[0]))

                   
        
