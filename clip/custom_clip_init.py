
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import load, tokenize
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from data.imagnet_prompts import imagenet_classes
from data.fewshot_datasets import fewshot_datasets
from data.office_home_prompts import office_home_classes
from data.cls_to_names import *

_tokenizer = _Tokenizer()

DOWNLOAD_ROOT='~/.cache/clip'

class ClipImageEncoder(nn.Module):
    def __init__(self, device, arch="ViT-L/14", image_resolution=224, n_class=1000):
        super(ClipImageEncoder, self).__init__()
        clip, embed_dim, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.encoder = clip.visual
        del clip.transformer
        torch.cuda.empty_cache()
        
        self.cls_head = nn.Linear(embed_dim, n_class)
    
    @property
    def dtype(self):
        return self.encoder.conv1.weight.dtype

    def forward(self, image):
        x = self.encoder(image.type(self.dtype))
        output = self.cls_head(x)
        return output


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, batch_size=None, n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size

        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)

        if ctx_init:
            # use given words to initialize context vectors
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None: 
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  #(N, L, D)
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors) # to be optimized

        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors) # to be optimized

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames

    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors) # to be optimized
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)

        clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)

        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

  

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim = 1,
        )

        return prompts
    
    def forward(self,initc, init=None):
            # the init will be used when computing CLIP directional loss
            if init is not None:
                ctx = init
            else:
                ctx = self.ctx
            text_initc = initc.cuda()

           
            # elif not ctx.size()[0] == self.n_cls:
            #     ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

            prefix = self.token_prefix
            suffix = self.token_suffix
            if self.batch_size is not None: 
                # This way only works for single-gpu setting (could pass batch size as an argument for forward())
                prefix = prefix.repeat(self.batch_size, 1, 1, 1)
                suffix = suffix.repeat(self.batch_size, 1, 1, 1)

            if self.learned_cls:
                assert self.class_token_position == "end"

            ctx_shifted = ctx.unsqueeze(0)    
            prompts_init = []
            for ctx_shifted_i in ctx_shifted:
                ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
                pro_cls = text_initc.unsqueeze(1).expand(-1, 4, -1).float()
                shift_pro_ctx =  ctx_i + pro_cls
                pts_i = self.construct_prompts(shift_pro_ctx, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
                prompts_init.append(pts_i)

            prompts_init = torch.stack(prompts_init)
            prompts_init = prompts_init.squeeze(0)
            

            return prompts_init


class ClipTestTimeTuning(nn.Module):
    def __init__(self, device, classnames, batch_size, criterion='cosine', arch="ViT-L/14",
                        n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        super(ClipTestTimeTuning, self).__init__()
        clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        # prompt tuning
        self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls)
        self.criterion = criterion
        
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)

    def get_text_features(self,initc):
        text_features = []
        prompts = self.prompt_learner(initc)
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        t_features = self.text_encoder(prompts, tokenized_prompts)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)

        return torch.mean(text_features, dim=0)

    def inference(self, image,initc):
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))

        text_features = self.get_text_features(initc)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits,image_features,text_features

    def forward(self, input,initc):
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            return self.inference(input,initc)


def get_coop(clip_arch, test_set, device, n_ctx, ctx_init, learned_cls=False):
    if test_set in fewshot_datasets:
        classnames = eval("{}_classes".format(test_set.lower()))
    elif test_set == 'bongard':
        if learned_cls:
            classnames = ['X', 'X']
        else:
            classnames = ['True', 'False']
    else:
        classnames = imagenet_classes

    model = ClipTestTimeTuning(device, classnames, None, arch=clip_arch,
                            n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls)

    return model

def get_coop_oh(clip_arch, test_set, device, n_ctx, ctx_init, learned_cls=False):
    if test_set in fewshot_datasets:
        classnames = eval("{}_classes".format(test_set.lower()))
    elif test_set == 'bongard':
        if learned_cls:
            classnames = ['X', 'X']
        else:
            classnames = ['True', 'False']
    else:
        classnames = office_home_classes

    model = ClipTestTimeTuning(device, classnames, None, arch=clip_arch,
                            n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls)

    return model