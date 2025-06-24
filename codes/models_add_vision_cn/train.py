# -*- coding: utf-8 -*-
# @Time    : 2022/11/18 15:21
# @Author  : Yao Linli

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, AdamW
from modules import MultiModalBartForConditionalGeneration
import numpy as np
import time
import os
import sys
import argparse
from collections import Counter
import random
random.seed(1234)
sys.path.append('../')
sys.path.append('./')
sys.path.append('../data_process/get_videoFeat_tsv/utils/')
from tsv_file import TSVFile
import base64
from utils.log import Logger
from utils.functions import set_seed, convert_continues_to_discrete
import pdb
import json
from tqdm import tqdm
from infer import generate, SELFBARTDataset_infer
import copy

def read_json(path):
    data = []
    with open(path, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            data.append(json.loads(line))
    return data


class SELFBARTDataset(Dataset):
    """
    this dataset is for training/validation/testing with the cross entropy loss.
    """

    def __init__(self, args, dataset, mode, tokenizer, use_command, use_reference,
                 max_example_len, max_frame_len=30, add_space=0, add_attr=1, dataset_percent=1):
        self.dataset = dataset
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_example_len = max_example_len # 60
        self.data = read_json(f'{dataset}/{mode}.json')
        self.data_size = len(self.data)
        self.max_frame_len = max_frame_len 
        self.visual_root = args.visual_root.format(mode) # "videoFeat_fps1_blip_cls_1000.tsv"
        self.val_num = args.val_num
        self.vfeat_dim = args.vfeat_dim
        self.type_embedding = args.type_embedding
        print(f'mode set size:{self.data_size}') 
        
        # video feature tsv file
        self.video_tsv_file = self.visual_root + ".tsv"
        self.video_lineidx_file = self.visual_root + ".lineidx"
        self.video_tsv = self.get_tsv_file(self.video_tsv_file)
        self.vid2tsvidx = self.prepare_video_key_to_index(self.video_tsv)
        
        self.vid_list = []
        self.Nframes_list = []
        self.word_tokenization_list = []
        self.definition_tokenization_list = []
        self.example_tokenization_list = []


        if add_space:
            data_dict_path = f'{dataset}/{mode}_mmfeatures_add_space.pt'
        else:
            data_dict_path = f'{dataset}/{mode}_mmfeatures.pt'
        # Get special tokens, which we added to the tokenizer
        special_token_dict = tokenizer.get_added_vocab()
        print(special_token_dict)
        self.special_token_dict = special_token_dict
        self.command_token_id = special_token_dict['<command>']
        self.reference_token_id = special_token_dict['<reference>']
        
        if os.path.exists(data_dict_path):
            # read all data sample
            print(f'Loading data from  {data_dict_path}.')
            data_dict = torch.load(data_dict_path)
            self.word_tokenization_list = data_dict['word_tokenization_list'] # [set size, token_ids] 
            self.definition_tokenization_list = data_dict['definition_tokenization_list'] # e.g. 1138316, [0, 5, 1318, 9, 145, 80, 12, 23944, 2]
            self.example_tokenization_list = data_dict['example_tokenization_list']
            self.vid_list = data_dict['vid_list']
            self.Nframes_list = data_dict['Nframes_list']
        else:
            # convert caption tokens to ids
            print(f'Tokenize the {mode} set.')
            # tokenize the command, lemma, reference, example
            for jterm in tqdm(self.data):
                vid, command, reference, newcap, dtype = jterm["vid"], jterm["command"], jterm["reference"], jterm["newcap"], jterm["dtype"]
                if reference is None:
                    reference = ''
                if add_space:
                    command = ' ' + command
                    reference = ' ' + reference
                    newcap = ' ' + newcap
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
                # pdb.set_trace()
                ids = tokenizer.encode(command, add_special_tokens=False) 
                self.word_tokenization_list.append(ids)
                
                mask = '[MASK]'
                if args.pos_type == 'self':
                    mask = '<mymask>'
                if '<mask>' in reference:
                        reference = reference.replace('<mask>', mask)
                ids = tokenizer.encode(reference, add_special_tokens=True)
                self.definition_tokenization_list.append(ids)

                ids = tokenizer.encode(newcap, add_special_tokens=True)
                self.example_tokenization_list.append(ids)

                vfeat_path = os.path.join(self.visual_root, str(vid)+".npy")
                self.vid_list.append(vfeat_path)
                Nframes = self.max_frame_len
                self.Nframes_list.append(Nframes)

            data_dict = {
                'vid_list': self.vid_list,
                'Nframes_list': self.Nframes_list,
                'word_tokenization_list': self.word_tokenization_list,
                'definition_tokenization_list': self.definition_tokenization_list,
                'example_tokenization_list': self.example_tokenization_list,
            }
            torch.save(data_dict, data_dict_path)

        # pdb.set_trace()
        # create encoder inputs
        self.encoder_input_list = []
        self.encoder_vtoken_list = []
        self.vfeat_path_list = []
        for i, (command, reference, vfeat_path, Nframes) in enumerate(zip(self.word_tokenization_list, \
                                                            self.definition_tokenization_list, self.vid_list, self.Nframes_list)):
            # (command, reference) newcap:
            # ([132, 495],  [0, 5, 1318, 9, 145, 80, 12, 23944, 2])
            encoder_input = []
            if use_command:
                encoder_input += [special_token_dict['<command>']] + command

            if use_reference:
                encoder_input += [special_token_dict['<reference>']] + reference
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
        # use parts of the training data to train the model
        if dataset_percent < 1 and self.mode == 'training':  
            sampled_index = np.random.choice(np.arange(self.len), int(self.len * dataset_percent))
            encoder_input_list = []
            encoder_vtoken_list = []
            example_tokenization_list = []
            example_vpath_list = []
            for i in sampled_index:
                encoder_input_list.append(self.encoder_input_list[i])
                encoder_vtoken_list.append(self.encoder_vtoken_list[i])
                example_tokenization_list.append(self.example_tokenization_list[i])
                example_vpath_list.append(self.vfeat_path_list[i])
            self.encoder_input_list = encoder_input_list
            self.example_tokenization_list = example_tokenization_list
            self.vfeat_path_list = example_vpath_list
            self.encoder_vtoken_list = encoder_vtoken_list

            self.len = len(self.encoder_input_list)
            print(f'Using {self.len} data instances to train the model.')
        elif self.val_num > 0 and self.mode != "training":
            num = min(self.len, self.val_num)
            sampled_index = random.sample(list(range(self.len)), num)
            self.encoder_input_list = [self.encoder_input_list[i] for i in sampled_index]
            self.example_tokenization_list = [self.example_tokenization_list[i] for i in sampled_index]
            self.vfeat_path_list = [self.vfeat_path_list[i] for i in sampled_index]
            self.encoder_vtoken_list = [self.encoder_vtoken_list[i] for i in sampled_index]


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
        # get vid feats
        vid = self.vfeat_path_list[idx].split('/')[-1].split('.npy')[0]
        tsvVid, cls_feats = self.get_row_from_tsv(self.video_tsv, vid) # [N, 768]
        assert tsvVid == vid
        Nframes = min(cls_feats.shape[0], self.max_frame_len)
        # randomly sample N frames in the cls_feats
        if cls_feats.shape[0] > Nframes:
            sample_idxs = np.sort(np.random.choice(np.arange(cls_feats.shape[0]), Nframes, replace=False))
            cls_feats = cls_feats[sample_idxs] 
        return torch.tensor(self.encoder_input_list[idx], dtype=torch.long), \
               torch.tensor(self.example_tokenization_list[idx], dtype=torch.long), \
               torch.tensor(self.encoder_vtoken_list[idx], dtype=torch.long), \
               torch.tensor(cls_feats.copy(), dtype=torch.float32), \
               torch.tensor([Nframes], dtype=torch.long)

    def __len__(self):
        return self.len

    def create_mini_batch(self, samples):
        encoder_input_list = {}
        encoder_input_list["text"] = [s[0] for s in samples]
        vision_input = [s[2] for s in samples]
        Nframes = [s[4][0] for s in samples]
        encoder_input_list["vision"] = [vision_input[i][:n+1] for i,n in enumerate(Nframes)]
        vision_features = [s[3] for s in samples]
        decoder_input_list = [s[1][:-1] for s in samples]
        decoder_label_list = [s[1][1:] for s in samples]
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
        visual_type_ids = torch.zeros(encoder_inputs["vision"].shape, dtype=torch.long)
        text_type_ids = torch.ones(encoder_inputs["text"].shape, dtype=torch.long)
        # update command type ids
        # pdb.set_trace()
        if self.type_embedding:
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
        # pdb.set_trace()
        decoder_inputs = pad_sequence(decoder_input_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        decoder_labels = pad_sequence(decoder_label_list, batch_first=True, padding_value=-100)
        return all_encoder_inputs, all_attention_mask, decoder_inputs, decoder_labels, vision_inputs, decoder_labels, all_type_ids

    def evaluate(self, model, local_rank, mode, dataloader):
        """
        compute the average loss over the test or validation set.
        :param model:
        :param local_rank:
        :param dataloader:
        :return:
        """
        datasize = self.len
        model.eval()
        total_loss = 0
        total_tokens = 0
        step = 0
        start = time.time()
        with torch.no_grad():
            for data in dataloader:
                data = [t.to(device) for t in data]
                encoder_inputs, attention_mask, decoder_inputs, decoder_labels, vision_features, attr_labels, type_ids = data
                if not args.type_embedding:
                    type_ids = None
                loss, logits = model(encoder_inputs, vision_features=vision_features, attention_mask=attention_mask,
                                     decoder_input_ids=decoder_inputs, labels=decoder_labels, type_ids=type_ids,
                                     output_attentions=True)[:2]
                if args.use_weighted_loss:
                    # only calculate attr-related loss
                    attr_loss, _ = model(encoder_inputs, vision_features=vision_features, attention_mask=attention_mask,
                                     decoder_input_ids=decoder_inputs, labels=attr_labels, type_ids=type_ids,
                                     output_attentions=True)[:2]
                    loss = args.alpha*loss + args.beta*attr_loss
                bts = encoder_inputs.shape[0]
                num_tokens = torch.sum(decoder_labels != -100)
                total_loss += loss * num_tokens
                total_tokens += num_tokens
                step += bts
                if local_rank in [-1, 0]:
                    print(
                        f'\r   Evaluating on the {mode} set for {step}/{datasize / torch.cuda.device_count()} '
                        f'takes {time.time() - start:.1f} seconds.', end='')

            if torch.cuda.device_count() > 1:
                torch.distributed.all_reduce_multigpu([total_loss])
            total_loss = total_loss.item()

            average_loss = total_loss / total_tokens
            used_time = time.time() - start
            print()
        model.train()
        return average_loss, used_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Controllable dictionary example generation.")
    parser.add_argument('--dataset', type=str, default='vatex', help='the path of the dataset.')
    parser.add_argument('--dataset_percent', type=float, default=1, help='The percentage of data used to train the model.')
    parser.add_argument('--val_num', type=int, default=-1, help='The number of validation data samples used to speed up the evaluation.')
    parser.add_argument('--infer_num', type=int, default=-1, help='The number of validation data samples used to speed up the evaluation.')
    
    parser.add_argument('--initialization', type=str, default='bart-base',
                        help='initialize the model with random values, bart-base or bart-large.')
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--test_batch_size', type=int, default=80)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-5, help='The initial learning rate for training.')
    parser.add_argument('--train', type=int, default=1, choices=[0, 1], help='1 for training, 0 for testing.')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--gpu', type=str, default='1', help='The ids of gpus for training.')
    parser.add_argument('--load_ckpt', type=str, default='', help='load pre-trained checkpoint (true or specific path)')
    # parser.add_argument('--model_path', type=str, default='', help='model path to save checkpoints')
    parser.add_argument('--num_workers', type=int, default=10, help='num_workers in dataloader')
    parser.add_argument('--data_root', type=str, default='data', help='the root path of dataset')
    parser.add_argument('--exp_name', type=str, default='', help='suffix name to distinguish different experiments')
    parser.add_argument('--visual_root', type=str, default='../data/emmad-edit/E-MMAD_fps1_clip_cn_cls_{}', help='root path of the visual features')
    parser.add_argument('--vfeat_dim', type=int, default=512, help='feature dim of extracted visual features')
    parser.add_argument('--type_embedding', type=int, default=0, help='using type embeddings to distince different types of tokens')
    # features
    parser.add_argument('--add_space', type=int, default=1, choices=[0, 1],
                        help='Whether add a space before the example, command and lemma '
                             'so that the tokens of the command do appear in the token sequence of the example.')
    parser.add_argument('--use_attr', type=int, default=0, choices=[0, 1],
                        help='whether use the specifical attr words as the input of the encoder.')
    
    parser.add_argument('--pos_type', type=str, default='aligned', choices=['aligned', 'self'])

    parser.add_argument('--use_command', type=int, default=1, choices=[0, 1],
                        help='whether use the command as the input of the encoder.'
                             '0 for no; 1 for yes')
    parser.add_argument('--use_reference', type=int, default=1, choices=[0, 1],
                        help='whether use the pos as the input of the encoder.')
    parser.add_argument('--num_bins', type=int, default=40,
                        help='the number of bins for lexical complexity features.')
    parser.add_argument('--max_example_len', type=int, default=150,
                        help='the max length of the dictionary examples.')
    # loss weight
    parser.add_argument('--use_weighted_loss', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='hyperparameter: ratio of CE loss(all tokens)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='hyperparameter: ratio of CE loss(attribute-related tokens)')
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

    prefix = f'lr_{args.lr}'
    exp_name = args.exp_name
    if args.add_space:
        prefix += f"_add_space"

    if args.use_attr:
        prefix += f"_add_attr"
        
    if args.dataset_percent < 1:
        prefix += f'_data_percent_{args.dataset_percent}'

    if args.use_command:
        prefix += '_use_command'
    else:
        pass
    
        
    if args.use_reference == 1:
        prefix += '_use_ref'

    model_path = f'../checkpoints_vision_cn/{args.dataset}_{args.initialization}{exp_name}/{prefix}'
    log_path = f'../logs_vision_cn/{args.dataset}_{args.initialization}{exp_name}'
    output_path = f'../outputs_vision_cn/{args.dataset}_{args.initialization}{exp_name}_generate'
    args.dataset = f'../{args.data_root}/{args.dataset}'
    if args.local_rank in [-1, 0]:
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_file = '{}/{}.log'.format(log_path, prefix)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)  
        args.model_path = model_path
        args.log_file = log_file
        logger = Logger(log_file)
        logger.logger.info(f'The log file is {log_file}.')
        logger.logger.info(f'The checkpoint path is {model_path}.')
        logger.logger.info(args)
        if args.train:
            logger.logger.info('Use {} gpus to train the model.'.format(args.n_gpu))
        else:
            logger.logger.info('Use {} gpus to evaluate the model.'.format(args.n_gpu))

    if len(args.load_ckpt) > 0:
        # load the pre-trained model and tokenizer
        tokenizer = BertTokenizer.from_pretrained(args.load_ckpt)
        model = MultiModalBartForConditionalGeneration.from_pretrained(args.load_ckpt)
        if args.local_rank in [-1, 0]:
            logger.logger.info('Initialize MultiModalBartForConditionalGeneration from checkpoint {}.'.format(args.load_ckpt))
    else:
        if args.initialization == "bart-base":
            try:
                tokenizer = BertTokenizer.from_pretrained('fnlp/bart-base-chinese')
                model = MultiModalBartForConditionalGeneration.from_pretrained('fnlp/bart-base-chinese')
            except:
                model = MultiModalBartForConditionalGeneration.from_pretrained('./bart-base-chinese')
                tokenizer = BertTokenizer.from_pretrained('./bart-base-chinese')
        else:
            tokenizer = BertTokenizer.from_pretrained(f'fnlp/{args.initialization}')
            model = MultiModalBartForConditionalGeneration.from_pretrained(f'fnlp/{args.initialization}')

        if args.local_rank in [-1, 0]:
            logger.logger.info(f'Initialize MultiModalBartForConditionalGeneration with default parameters {args.initialization}.')
        # add special tokens to the vocabulary
        special_tokens = []
        # add <command>, <lemma>, <pos>, <reference>
        special_tokens.append('<video>')
        special_tokens.append('<v>')
        special_tokens.append('<command>')
        special_tokens.append('<reference>')
        special_tokens.append('<add>')
        special_tokens.append('<dele>')
        special_tokens.append('<keep>')
        special_tokens.append('<null>')
        special_tokens.append('<attr>')
        special_tokens.append('<mymask>')
        special_tokens.append('<mask>')

        special_tokens_dict = {
            'additional_special_tokens': special_tokens,
        }
   
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        if args.local_rank in [-1, 0]:
            print(f'We have added {num_added_toks} special tokens to the vocabulary: {tokenizer.get_added_vocab()}.')
            print(f"The original vocabulary size is {tokenizer.vocab_size}; "
                  f"the extended vocabulary size is {len(tokenizer)}.")

        # randomly initialize the newly added special tokens.
        # see https://huggingface.co/transformers/main_classes/model.html for details
        model.resize_token_embeddings(len(tokenizer))
    if args.local_rank == -1 or args.n_gpu <= 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'device {device}.')
    else:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        print(f'local rank: {args.local_rank}, device {device}.')
    model = model.to(device)

    # train
    if args.train == 1:
        # 
        training_set = SELFBARTDataset(args, args.dataset, "training", tokenizer, args.use_command, args.use_reference, args.max_example_len, 
                                   add_space=args.add_space, add_attr=args.use_attr, dataset_percent=args.dataset_percent)
        validation_set = SELFBARTDataset(args, args.dataset, "validation", tokenizer, args.use_command, args.use_reference, args.max_example_len, 
                                     add_space=args.add_space, add_attr=args.use_attr)
        if args.local_rank in [-1, 0]:
            logger.logger.info(f'The size of the training set is {len(training_set)}; '
                               f'the size of the validation set is {len(validation_set)}.')
        if args.local_rank == -1 or args.n_gpu <= 1:
            training_sampler = torch.utils.data.RandomSampler(training_set)
            validation_sampler = torch.utils.data.SequentialSampler(validation_set)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
            training_sampler = torch.utils.data.distributed.DistributedSampler(training_set)
            validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_set)
        training_dataloader = DataLoader(training_set, num_workers=args.num_workers, batch_size=args.batch_size,
                                         sampler=training_sampler, collate_fn=training_set.create_mini_batch)
        validation_dataloader = DataLoader(validation_set, num_workers=args.num_workers, batch_size=args.test_batch_size,
                                           sampler=validation_sampler, collate_fn=training_set.create_mini_batch)
    # test
    else:
        test_set = SELFBARTDataset(args, args.dataset, "test", tokenizer, args.use_command, args.use_reference, args.max_example_len,
                               add_space=args.add_space, add_attr=args.use_attr)
        if args.local_rank in [-1, 0]:
            logger.logger.info(f'The size of the test set is {len(test_set)}.')
        if args.local_rank == -1 or args.n_gpu <= 1:
            test_sampler = torch.utils.data.SequentialSampler(test_set)
        else:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
        test_dataloader = DataLoader(test_set, num_workers=args.num_workers, batch_size=args.test_batch_size,
                                     sampler=test_sampler, collate_fn=test_set.create_mini_batch)
   
    # test
    if args.train == 0:
        average_loss, used_time = test_set.evaluate(model, args.local_rank, 'test', test_dataloader)
        if args.local_rank in [-1, 0]:
            logs = f'   Evaluate on the test set: average loss {average_loss:.3f}, ' \
                   f' taking {used_time:.1f} seconds.\n'
            logger.logger.info(logs)
    # train
    else:
        average_loss, used_time = validation_set.evaluate(model, args.local_rank, 'validation', validation_dataloader)
        optimizer = AdamW(model.parameters(), lr=args.lr)
       
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2, verbose=True,
                                                               min_lr=1e-6)  
        scheduler.step(average_loss) # # Reduce learning rate when average_val_loss has stopped improving
        best_loss = average_loss
        if args.local_rank in [-1, 0]:
            logs = f'   Evaluate on the validation set: average loss {average_loss:.3f}, ' \
                   f' taking {used_time:.1f} seconds.\n'
            logger.logger.info(logs)
        evaluate_steps = int(len(training_set) / args.batch_size / 3)
        print_steps = 10
        global_steps = 0
        local_step = 0
        total_loss = 0
        start = time.time()
        # fine-tune bart on the training dataset
        for epoch in range(args.epochs):
            for i, data in enumerate(training_dataloader):
                global_steps += 1
                local_step += 1
                data = [t.to(device) for t in data]
                encoder_inputs, attention_mask, decoder_inputs, decoder_labels, vision_features, attr_labels, type_ids = data
                if not args.type_embedding:
                    type_ids = None
                loss, logits = model(encoder_inputs, vision_features=vision_features, attention_mask=attention_mask,
                                     decoder_input_ids=decoder_inputs, labels=decoder_labels, type_ids=type_ids,
                                     output_attentions=True)[:2]
                if args.use_weighted_loss:
                    # only calculate attr-related loss
                    attr_loss, _ = model(encoder_inputs, vision_features=vision_features, attention_mask=attention_mask,
                                     decoder_input_ids=decoder_inputs, labels=attr_labels, type_ids=type_ids,
                                     output_attentions=True)[:2]
                    loss = args.alpha*loss + args.beta*attr_loss
                # zero the parameter gradients
                optimizer.zero_grad()
                # backward
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if global_steps % print_steps == 0 and args.local_rank in [-1, 0]:
                    print("\rEpoch {}/{}, {}/{}, global steps {}, average loss is {:.3f}, "
                          " {} steps uses {:.1f} seconds.".format(epoch + 1, args.epochs, i + 1, len(training_dataloader),
                                                                  global_steps, total_loss / local_step,
                                                                  local_step, time.time() - start), end='')
                if global_steps % evaluate_steps == 0:
                    if args.local_rank in [-1, 0]:
                        print()
                    average_loss, used_time = validation_set.evaluate(model, args.local_rank, 'validation',
                                                                      validation_dataloader)
                    if args.local_rank in [-1, 0]:
                        logs = f'   Evaluate on the validation set: average loss {average_loss:.3f}, ' \
                               f' taking {used_time:.1f} seconds.'
                        logger.logger.info(logs)
                    if average_loss < best_loss:
                        best_loss = average_loss
                        if args.local_rank in [-1, 0]:
                            logger.logger.info('Save the model at {}.'.format(args.model_path))
                            # Simple serialization for models and tokenizers
                            model_to_save = model.module if hasattr(model, "module") else model
                            model_to_save.save_pretrained(args.model_path)
                            tokenizer.save_pretrained(args.model_path)

                    logger.logger.info('')
                    scheduler.step(average_loss)
                    start = time.time()
                    total_loss = 0
                    local_step = 0
                    
        # inference: test generate
        # update model to the best checkpoint
        tokenizer = BertTokenizer.from_pretrained(args.model_path)
        model = MultiModalBartForConditionalGeneration.from_pretrained(args.model_path)
        model = model.to(device)
        model.eval()
        
        mode = "test"
        special_token_dict = tokenizer.get_added_vocab()
        test_set = SELFBARTDataset_infer(args, args.dataset, mode, tokenizer, args.use_command, args.use_reference, 
                                add_space=args.add_space, add_attr=args.use_attr, expected_len=-2, val_num=args.infer_num, special_token_dict=special_token_dict)
        
        
        test_sampler = torch.utils.data.SequentialSampler(test_set)
        test_dataloader = DataLoader(test_set, num_workers=8, batch_size=args.batch_size,
                                    sampler=test_sampler, collate_fn=test_set.create_mini_batch)


        logger.logger.info(f'The size of the {mode} set is {len(test_set)}.')
        

        output_file = f'{output_path}/{mode}_{prefix}.json'
        output_ckpt_file = f'{model_path}/'
        
        with torch.no_grad(), open(output_file, 'w') as fw:
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
                    # if len(generated_example) == 0:
                    #     pdb.set_trace()
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
            copy_to_ckpt = ["cp", "-r", output_file, output_ckpt_file]
            os.system(" ".join(copy_to_ckpt))
