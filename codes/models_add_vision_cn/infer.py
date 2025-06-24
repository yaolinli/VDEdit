# -*- coding: utf-8 -*-
# @Time    : 2022/11/18 15:21
# @Author  : Yao Linli

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import os
import sys
import argparse
from transformers import BertTokenizer
from modules import MultiModalBartForConditionalGeneration
sys.path.append('../')
sys.path.append('../data_process/get_videoFeat_tsv/utils/')
from tsv_file import TSVFile
import base64
from utils.log import Logger
from utils.functions import set_seed, convert_continues_to_discrete
import pdb
from tqdm import tqdm
import json
import random
import copy
random.seed(1234)


def read_json(path):
    data = []
    with open(path, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            data.append(json.loads(line))
    return data

class SELFBARTDataset_infer(Dataset):
    """
    this dataset is used to generate dictionary examples with the test/validation set without any loss.
    """
    def __init__(self, args, dataset, mode, tokenizer, use_command, use_reference, 
                max_frame_len=30, expected_len=-1, add_space=0, add_attr=1, val_num=-1, special_token_dict=None):
        self.dataset = dataset
        self.mode = mode
        self.tokenizer = tokenizer
        self.data = read_json(f'{dataset}/{mode}.json')
        self.data_size = len(self.data)
        self.max_frame_len = max_frame_len 
        self.visual_root = args.visual_root.format(mode) # "videoFeat_fps1_blip_cls_1000.tsv"
        self.val_num = val_num
        self.vfeat_dim = args.vfeat_dim
        print(f'mode set size:{self.data_size}')
        
        # video feature tsv file
        self.video_tsv_file = self.visual_root + ".tsv"
        self.video_lineidx_file = self.visual_root + ".lineidx"
        self.video_tsv = self.get_tsv_file(self.video_tsv_file)
        self.vid2tsvidx = self.prepare_video_key_to_index(self.video_tsv)
        self.special_token_dict = special_token_dict
        self.command_token_id = special_token_dict['<command>']
        self.reference_token_id = special_token_dict['<reference>']
        
        self.vid_list = []
        self.Nframes_list = []
        self.attr_list = []
        self.vname_list = []
        self.word_list = []
        self.definition_list = []
        self.example_list = []
        self.dtype_list = []
        self.oldcap_list = []

        self.word_tokenization_list = []
        self.definition_tokenization_list = []
        self.example_tokenization_list = []

        # load txt, which is used for the output
        if add_space:
            data_dict_path = f'{dataset}/{mode}_mmfeatures_add_space.pt'
        else:
            data_dict_path = f'{dataset}/{mode}_mmfeatures.pt'

        for jterm in self.data:
            vid, command, definition, example, attr = jterm["vid"], jterm["command"], jterm["reference"], jterm["newcap"], (jterm["attr"], jterm["atype"])
            oldcap = jterm["oldcap"]
            if "dtype" in jterm:
                dtype = jterm["dtype"]
            else:
                dtype = None
            self.vname_list.append(vid)
            self.word_list.append(command)
            self.definition_list.append(definition)
            self.example_list.append(example)
            self.attr_list.append(attr)
            self.dtype_list.append(dtype)
            self.oldcap_list.append(oldcap)
            
        if os.path.exists(data_dict_path):
            # read all data sample
            print(f'Loading data from  {data_dict_path}.')
            data_dict = torch.load(data_dict_path)
            self.word_tokenization_list = data_dict['word_tokenization_list'] # [set size, token_ids] 
            self.definition_tokenization_list = data_dict['definition_tokenization_list'] # e.g. 1138316, [0, 5, 1318, 9, 145, 80, 12, 23944, 2]
            #self.example_tokenization_list = data_dict['example_tokenization_list']
            self.vid_list = data_dict['vid_list']
            self.Nframes_list = data_dict['Nframes_list']
        else:
            # convert caption tokens to ids
            print(f'Tokenize the {mode} set.')
            # tokenize the command, lemma, definition, example
            for jterm in tqdm(self.data):
                vid, command, definition, example = jterm["vid"], jterm["command"], jterm["reference"], jterm["newcap"]
                if add_space:
                    command = ' ' + command
                    definition = ' ' + definition
                    # example = ' ' + example
                if add_attr:
                    # pdb.set_trace()
                    attr, atype = jterm["attr"], jterm["atype"]
                    if attr is None:
                        attr = ""
                    command = command + ' <attr> ' + attr
                else:
                    command = command + ' <attr> '
                # # convert tokens to ids
                # e.g. ['<s>', 'Ġthe', 'Ġquality', 'Ġof', 'Ġbeing', 'Ġtwo', '-', 'dimensional', '</s>']
                # -> [0, 0, 5, 1318, 9, 145, 80, 12, 23944, 2, 2]
                ids = tokenizer.encode(command, add_special_tokens=False) 
                self.word_tokenization_list.append(ids)

                # if '<mask>' in definition:
                #     definition = definition.replace('<mask>', '[MASK]')
                
                mask = '[MASK]'
                if args.pos_type == 'self':
                    mask = '<mymask>'
                if '<mask>' in definition:
                    definition = definition.replace('<mask>', mask)
                ids = tokenizer.encode(definition, add_special_tokens=True)
                self.definition_tokenization_list.append(ids)

                # ids = tokenizer.encode(example, add_special_tokens=True)
                # self.example_tokenization_list.append(ids)
                
                vfeat_path = os.path.join(self.visual_root, str(vid)+".npy")
                self.vid_list.append(vfeat_path)
                Nframes = self.max_frame_len
                self.Nframes_list.append(Nframes)

            data_dict = {
                'vid_list': self.vid_list,
                'Nframes_list': self.Nframes_list,
                'word_tokenization_list': self.word_tokenization_list,
                'definition_tokenization_list': self.definition_tokenization_list,
                #'example_tokenization_list': self.example_tokenization_list,
            }
            torch.save(data_dict, data_dict_path)
            
        # create encoder inputs
        self.encoder_input_list = []
        self.encoder_vtoken_list = []
        self.vfeat_path_list = []
        for i, (command, definition, vfeat_path, Nframes) in enumerate(zip(self.word_tokenization_list, \
                                                            self.definition_tokenization_list, self.vid_list, self.Nframes_list)):
            # (command, definition) example:
            # ([132, 495],  [0, 5, 1318, 9, 145, 80, 12, 23944, 2])
            encoder_input = []
            if use_command:
                encoder_input += [special_token_dict['<command>']] + command

            if use_reference:
                encoder_input += [special_token_dict['<reference>']] + definition
            self.encoder_input_list.append(encoder_input)
            self.vfeat_path_list.append(vfeat_path)
            # video token id
            # example: <video> <v> <v> <v> ....
            # each video has N(frames=N) tokens
            encoder_vtoken = []
            encoder_vtoken += [special_token_dict['<video>']]
            if Nframes > self.max_frame_len:
                Nframes = self.max_frame_len
            encoder_vtoken += [special_token_dict['<v>']] * Nframes
            self.encoder_vtoken_list.append(encoder_vtoken)

        self.len = len(self.encoder_input_list)

        if self.val_num > 0 and self.mode != "training":
            num = min(self.len, self.val_num)
            sampled_index = random.sample(list(range(self.len)), num)
            self.encoder_input_list = [self.encoder_input_list[i] for i in sampled_index]
            self.encoder_vtoken_list = [self.encoder_vtoken_list[i] for i in sampled_index]
            self.vfeat_path_list = [self.vfeat_path_list[i] for i in sampled_index]
            self.len = len(self.encoder_input_list)
            print(f'Using {self.len} data instances to eval the model.')      

    def prepare_video_key_to_index(self, tsv):
        return {tsv.get_key(i) : i for i in range(tsv.num_rows())}

    def get_tsv_file(self, tsv_file):
        if tsv_file:
            return TSVFile(tsv_file)

    def get_row_from_tsv(self, tsv, vid):
        tsv_idx = self.vid2tsvidx[vid]
        row = tsv[tsv_idx]
        feat_info = json.loads(row[1])
        vid = row[0]
        if "cn" in self.visual_root.split("/")[-1]:
            dtype = np.float32
        else:
            dtype = np.float16
        feat = np.frombuffer(base64.b64decode(feat_info["feature"]), dtype).reshape(-1, self.vfeat_dim)
        return vid, feat
    
    def __getitem__(self, idx):
        vid = self.vfeat_path_list[idx].split('/')[-1].split('.npy')[0]
        tsv_vid, cls_feats = self.get_row_from_tsv(self.video_tsv, vid) # [N, 768]
        assert tsv_vid == vid
        Nframes = min(cls_feats.shape[0], self.max_frame_len)
        cls_feats = cls_feats[:Nframes] 
        return  torch.tensor(self.encoder_input_list[idx], dtype=torch.long), \
                torch.tensor(self.encoder_vtoken_list[idx], dtype=torch.long), \
                torch.tensor(cls_feats.copy(), dtype=torch.float32), \
                torch.tensor([Nframes], dtype=torch.long) 


    def __len__(self):
        return self.len

    def create_mini_batch(self, samples):
        encoder_input_list = {}
        encoder_input_list["text"] = [s[0] for s in samples]
        vision_input = [s[1] for s in samples]
        Nframes = [s[3][0] for s in samples]
        encoder_input_list["vision"] = [vision_input[i][:n+1] for i,n in enumerate(Nframes)]
        vision_features = [s[2] for s in samples]
        # Mask to avoid performing attention on padding token indices in encoder_inputs.
        _mask = {}
        attention_mask = {}
        encoder_inputs = {}
        for m in ["text", "vision"]:
            # _mask/attention_mask/encoder_inputs: shape = [batch size, max_text_len] e.g. torch.Size([10, 25])
            _mask[m] = pad_sequence(encoder_input_list[m], batch_first=True, padding_value=-100)
            attention_mask[m] = torch.zeros(_mask[m].shape, dtype=torch.float32)
            attention_mask[m] = attention_mask[m].masked_fill(_mask[m] != -100, 1)
            encoder_inputs[m] = pad_sequence(encoder_input_list[m], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        all_encoder_inputs = torch.cat((encoder_inputs["vision"], encoder_inputs["text"]), dim=1)
        all_attention_mask = torch.cat((attention_mask["vision"], attention_mask["text"]), dim=1)

        # update command type ids
        visual_type_ids = torch.zeros(encoder_inputs["vision"].shape, dtype=torch.long)
        text_type_ids = torch.ones(encoder_inputs["text"].shape, dtype=torch.long)
        bz = text_type_ids.shape[0]
        for i in range(bz):  
            text_seq = list(encoder_input_list["text"][i])
            command_pos = text_seq.index(self.command_token_id)
            reference_pos = text_seq.index(self.reference_token_id)
            text_type_ids[i, command_pos:reference_pos] += 1
        all_type_ids = torch.cat((visual_type_ids, text_type_ids), dim=1)

        # pad vision features
        # vision features --> batch size * [Nframes, dim], e.g. torch.Size([12, 768])
        max_vlen = encoder_inputs["vision"].shape[1] - 1
        dim = vision_features[0].shape[-1]
        vision_inputs = []
        for v in vision_features:
            padd_features = torch.zeros((max_vlen, dim), dtype=torch.float32)
            vlen = min(v.shape[0], max_vlen)
            padd_features[:vlen] = v[:vlen]
            vision_inputs.append(padd_features)
        vision_inputs = torch.stack(vision_inputs)
        return all_encoder_inputs, all_attention_mask, vision_inputs, all_type_ids

def contain(words, example):
    example = ' ' + example + ' '
    for i, word in enumerate(words):
        start = 0
        while True:
            start = example.find(word, start)
            if start!=-1:
                start_char = example[start-1]
                end_char = example[start+len(word)]
                if not start_char.isalnum() and not end_char.isalnum():
                    if i == 0:
                        return 1
                    return 2
                start += 1
            else:
                break
    return 0

def generate(args, model, tokenizer, word_list, input_ids = None, vision_features=None, attention_mask = None, type_ids=None, max_decoding_len=None, num_beams=1, repetition_penalty=1,
             top_k=50, top_p=0.9, decoder_chain = 1):
    """

    :param input_ids:
    :param attention_mask:
    :param max_decoding_len:
    :param num_beams:
    :param repetition_penalty:
    :param top_k:
    :param top_p:
    :param decoder_chain: run multiple parallel chains for top-k or top-p sampling, then choose the one contains the given word
    :return:
    """
    batch_size = input_ids.shape[0]
    if decoder_chain>1:
        input_ids = input_ids.repeat(decoder_chain,1,1).reshape(batch_size*decoder_chain, -1)
        attention_mask = attention_mask.repeat(decoder_chain,1,1).reshape(batch_size*decoder_chain, -1)

    # generate text until the output length (which includes the context length) reaches 50
    if args.decoding_strategy == 1:
        # model.generate() in /opt/conda/envs/CDEG/lib/python3.6/site-packages/transformers/generation_utils.py
        output = model.generate(input_ids=input_ids, vision_features=vision_features, attention_mask=attention_mask, type_ids=type_ids, max_length=max_decoding_len,
                                num_beams=num_beams, repetition_penalty=repetition_penalty, output_attentions = True)
    elif args.decoding_strategy == 2:
        output = model.generate(input_ids=input_ids, vision_features=vision_features, attention_mask=attention_mask, type_ids=type_ids, max_length=max_decoding_len,
                                do_sample=True, top_k=top_k, repetition_penalty=repetition_penalty, output_attentions = True)
    elif args.decoding_strategy == 3:
        output = model.generate(input_ids=input_ids, vision_features=vision_features, attention_mask=attention_mask, type_ids=type_ids, max_length=max_decoding_len,
                                do_sample=True, top_p=top_p, top_k=0,
                                repetition_penalty=repetition_penalty, output_attentions = True)

    generated_example_list = []
    selected_example_list = []

    for i in range(batch_size*decoder_chain):
        generated_example = tokenizer.convert_ids_to_tokens(output[i], skip_special_tokens=True)
        # generated_example = tokenizer.decode(output[i], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        generated_example_list.append(generated_example)
    # for e in generated_example_list:
    #     print(e)
    if decoder_chain>1:
        for i in range(batch_size):
            examples = [generated_example_list[j*batch_size+i]   for j in range(decoder_chain)]
            selected_example_list.append(examples[0])
            for example in examples:
                contain_code = contain([word_list[i]], example)
                if contain_code>0:
                    selected_example_list[i] = example
                    break
    else:
        selected_example_list = generated_example_list
    return selected_example_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Controllable dictionary example generation.")
    # hyper-parameters for well-trained models
    parser.add_argument('--dataset', type=str, default='oxford', help='the path of the dataset.')
    parser.add_argument('--dataset_percent', type=float, default=1,
                        help='The percentage of data used to train the model.')
    parser.add_argument('--initialization', type=str, default='bart-base',
                        choices=['bart-random-base', 'bart-base', 'bart-large'],
                        help='initialize the model with random values, bart-base or bart-large.')
    parser.add_argument('--test_mode', type=int, default=1, choices=[0, 1, 2], help='0 for validation, 1 for test set, ')
    parser.add_argument('--infer_num', type=int, default=-1, help='The number of validation data samples used to speed up the evaluation.')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-5, help='The initial learning rate for training.')
    parser.add_argument('--gpu', type=str, default='3', help='The ids of gpus for training.')
    parser.add_argument('--model_path', type=str, default=None, help='The path of checkpoint.')

    # hyper-parameters for features
    parser.add_argument('--add_space', type=int, default=1, choices=[0,1],
                        help='Whether add a space before the example, word and lemma '
                             'so that the tokens of the word do appear in the token sequence of the example.')
    parser.add_argument('--use_attr', type=int, default=0, choices=[0, 1],
                        help='whether use the specifical attr words as the input of the encoder.')
    parser.add_argument('--pos_type', type=str, default='aligned', choices=['aligned', 'self'])
    parser.add_argument('--use_command', type=int, default=1, choices=[0,1,2,3],
                        help='whether use the word or lemma as the input of the encoder.'
                             '0 for word; 1 for lemma; 2 for both word and lemma; 3 for not using both.')
    parser.add_argument('--use_reference', type=int, default=1, choices=[0, 1],
                        help='whether use the pos as the input of the encoder.')
    # parser.add_argument('--use_pos', type=int, default=0, choices=[0,1],
    #                     help='whether use the pos as the input of the encoder.')
    # parser.add_argument('--use_example_len', type=int, default=0, choices=[0,1],
    #                     help='whether use the length of examples as the input of the encoder.')
    parser.add_argument('--use_lexical_complexity', type=int, default=0, choices=[0,1,2,3,4,5,6],
                        help='0 denotes the lexical complexity is not used as the input of the encoder;'
                             '1 denotes the word_rank_lexical_complexity is regarded as the lexical complexity;'
                             '2 denotes the token_rank_lexical_complexity is regarded as the lexical complexity;'
                             '3 denotes the external_word_rank_lexical_complexity is regarded as the lexical complexity;'
                             '4 denotes the external_token_rank_lexical_complexity is regarded as the lexical complexity;'
                             '5 denotes the flesch_reading_ease is regarded as the lexical complexity;'
                             '6 denotes the flesch_kincaid_grade_level is regarded as the lexical complexity;')
    parser.add_argument('--num_bins', type=int, default=40,
                        help='the number of bins for lexical complexity features.')
    parser.add_argument('--data_root', type=str, default='data', help='the root path of dataset')
    parser.add_argument('--max_example_len', type=int, default= 150,
                        help='the max length of the dictionary examples.')
    parser.add_argument('--exp_name', type=str, default='', help='suffix name to distinguish different experiments')
    parser.add_argument('--visual_root', type=str, default='../data/emmad-edit/E-MMAD_fps1_clip_cn_cls_{}', help='root path of the visual features')
    parser.add_argument('--vfeat_dim', type=int, default=512, help='feature dim of extracted visual features')
    parser.add_argument('--type_embedding', type=int, default=0, help='using type embeddings to distince different types of tokens')
    # hyper-parameters for decoding strategy
    parser.add_argument('--decoding_strategy', type=int, default=1, choices=[1,2,3],
                        help='1 for greedy/beam search decoding; 2 for top-k decoding; 3 for top-p decoding')
    parser.add_argument('--num_beams', type=int, default=5,
                        help='1 for greedy decoding; '
                             'the number greater than 1 denotes beam search decoding.')
    parser.add_argument('--top_k', type=int, default=50,
                        help='The number of highest probability vocabulary tokens to keep for top-k-filtering. '
                             'Between 1 and infinity.')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='The cumulative probability of parameter highest probability vocabulary tokens to keep '
                             'for nucleus sampling. Must be between 0 and 1.')
    parser.add_argument('--decoder_chain', type=int, default=1,
                        help='the number of parallel chains for top-k or top-p, each chain refers to an unique token sequence.')
    parser.add_argument('--repetition_penalty', type=float, default=1.3,
                        help='Between 1.0 and infinity.1.0 means no penalty.Default to 1.0.')

    parser.add_argument('--max_decoding_len', type=int, default= 150,
                        help='the max length of the dictionary examples.')

    parser.add_argument('--expected_len', type=int, default= -2,
                        help='Specify the expected length of generated examples.'
                             '-2 denotes not using this token.'
                             '-1 denotes use the gold label of the validation/test set.'
                             'the value should be integer in [0, num_bins).')

    parser.add_argument('--expected_lexical_complexity', type=int, default= -2,
                        help='Specify the expected lexical complexity of generated examples.'
                             '-2 denotes not using this token.'
                             '-1 denotes use the gold label of the validation/test set.'
                             'the value should be integer in [1, max_example_len].')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.n_gpu = torch.cuda.device_count()
    set_seed(args.seed, args.n_gpu)

    if args.test_mode == 0:
        print('Evaluate the model on the validation set.')
        mode = 'validation'
    elif args.test_mode == 1:
        print('Evaluate the model on the test set.')
        mode = 'test'
    else:
        print('Evaluate the model on specified inputs.')
        mode = 'specified'
    if args.decoding_strategy ==1:
        args.decoder_chain=1

    prefix = f'lr_{args.lr}'

    if args.add_space:
        prefix += f"_add_space"

    if args.use_attr:
        prefix += f"_add_attr"
        
    if args.dataset_percent<1:
        prefix += f'_data_percent_{args.dataset_percent}'

    if args.use_command:
        prefix += '_use_command'

    if args.use_reference:
        prefix += '_use_ref'

    if args.model_path is None:
        model_path = f'../checkpoints_vision_cn/{args.dataset}_{args.initialization}{args.exp_name}/{prefix}'
    else:
        model_path = args.model_path


    prefix += f'_max_decoding_len_{args.max_decoding_len}'
    if args.decoding_strategy==1:
        if args.num_beams==1:
            prefix += f'_greedy'
        else:
            prefix += f'_beam_search_{args.num_beams}'
    elif args.decoding_strategy==2:
        prefix += f'_top_k_{args.top_k}_{args.decoder_chain}'
    elif args.decoding_strategy==3:
        prefix += f'_top_p_{args.top_p}_{args.decoder_chain}'
    else:
        raise ValueError('Please input the correct decoding strategy index (1, 2, or 3).')
    if args.repetition_penalty>1:
        prefix += f'_repetition_penalty_{args.repetition_penalty}'

    log_path = f'../logs_vision_cn/{args.dataset}_{args.initialization}{args.exp_name}_generate'
    output_path = f'../outputs_vision_cn/{args.dataset}_{args.initialization}{args.exp_name}_generate'

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    args.model_path = model_path

    args.output_file = f'{output_path}/{mode}_{prefix}.json'
    args.log_file = f'{log_path}/{mode}_{prefix}.log'

    logger = Logger(args.log_file)
    logger.logger.info(f'The log file is {args.log_file}.')
    logger.logger.info(f'The output file is {args.output_file}.')
    logger.logger.info(args)
    # load the pre-trained model and tokenizer
    logger.logger.info(f'Loading the model from checkpoint {args.model_path}.')
    args.dataset = f'../{args.data_root}/{args.dataset}'

    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    model = MultiModalBartForConditionalGeneration.from_pretrained(args.model_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device {device}.')
    model = model.to(device)
    model.eval()

    special_token_dict = tokenizer.get_added_vocab()
    
    
    test_set = SELFBARTDataset_infer(args, args.dataset, mode, tokenizer, args.use_command, args.use_reference, 
                            add_space=args.add_space, add_attr=args.use_attr, expected_len=args.expected_len, val_num=args.infer_num, special_token_dict=special_token_dict)
    if args.infer_num > 1:
        logger.logger.info(f'The size of the {mode} set is {min(len(test_set), args.infer_num)}.')
    else:
        logger.logger.info(f'The size of the {mode} set is {len(test_set)}.')
        
    test_sampler = torch.utils.data.SequentialSampler(test_set)
    test_dataloader = DataLoader(test_set, num_workers=8, batch_size=args.batch_size,
                                sampler=test_sampler, collate_fn=test_set.create_mini_batch)


    with torch.no_grad(), open(args.output_file, 'w') as fw:
        start = time.time()
        batch_index = -1
        for data in test_dataloader:
            batch_index += 1
            data = [t.to(device) for t in data]
            input_ids, attention_mask, vision_features, type_ids = data

            batch_size = input_ids.shape[0]
            word_list = []
            for index in range(args.batch_size*batch_index, args.batch_size*batch_index+batch_size):
                command = test_set.word_list[index]
                word_list.append(command)
            if not args.type_embedding:
                type_ids = None
            generated_examples = generate(args, model, tokenizer, word_list, input_ids=input_ids, vision_features=vision_features, attention_mask=attention_mask, type_ids=type_ids, 
                                            max_decoding_len=args.max_decoding_len, num_beams=args.num_beams,
                                            repetition_penalty=args.repetition_penalty,
                                            top_k=args.top_k, top_p=args.top_p, decoder_chain=args.decoder_chain)

            # write the output into the output file
            for i, j in zip(range(args.batch_size*batch_index, args.batch_size*batch_index+batch_size), range(batch_size)):
                index = i
                vid = test_set.vname_list[index]
                command = test_set.word_list[index]
                definition = test_set.definition_list[index]
                dtype = test_set.dtype_list[index]
                oldcap = test_set.oldcap_list[index]
                attr = test_set.attr_list[index]
                examples = test_set.example_list[index]
                generated_example = generated_examples[j]
                # pdb.set_trace()
                if generated_example[0] == ' ':  # remove the space of the begin
                    generated_example = generated_example[1:]

                result = {}
                result["vid"] = str(vid)
                result["oldcap"] = oldcap
                result["dtype"] = dtype
                result["command"] = command
                result["attr"] = attr[0]
                result["atype"] = attr[1]
                result["reference"] = definition
                result["newcap_gt_case"] = examples
                result["newcap_gt"] = tokenizer.convert_ids_to_tokens(tokenizer.encode(examples, return_tensors='pt')[0])
                result["newcap_gt"] = " ".join(result["newcap_gt"])
                result["newcap_generated_case"] = "".join(generated_example)
                result["newcap_generated"] = " ".join(generated_example)
                fw.write(json.dumps(result, ensure_ascii=False)+"\n")
            # break
            print(f'\rProcess {args.batch_size*(batch_index+1)}/{len(test_set)}, used {time.time()-start:.1f} seconds.', end='')

        logger.logger.info(f'The inference latency is {time.time()-start:.2f}\n')
        os.system("cp -r {} {}".format(args.output_file, args.model_path))



