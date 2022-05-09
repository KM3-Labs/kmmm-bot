from typing import Optional
from cogs.utils.tf.validation import GPTGenRequest, Response
import torch
import ray
from ray import serve

import logging
from .autohf import AutoHF
from .softprompt import SoftPrompt, AutoModelForSoftPromptLM, current_sp, resize_model_embeddings
from .tensorize import tensorize, untensorize
from .utils import get_dtype, tensorized_path
from .warpers import *
# from soft_prompt import SoftPrompt as SoftPromptModel
from transformers import (AutoConfig, AutoTokenizer,
                          LogitsProcessorList, MaxLengthCriteria,
                          MaxTimeCriteria, NoBadWordsLogitsProcessor,
                          StoppingCriteriaList, TemperatureLogitsWarper,
                          TopKLogitsWarper, TopPLogitsWarper, MinLengthLogitsProcessor)

from pathlib import Path

import numpy as np
import zlib

try:
    import transformers
    from quantization import GPTJBlock, GPTJForCausalLM
except ImportError:
    pass # don't do quantization


@serve.deployment(num_replicas=2, name="GPT", ray_actor_options={"num_gpus": 1})  # make resource and replica amount in config
class GPTHF(AutoHF):
    def __init__(self, model_name='hakurei/gpt-j-random-tinier', device=None, parallelize=False, quantized=False, tensorized=False):
        super().__init__(model_name=model_name, decoder=True)
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
        model_dtype = get_dtype(device)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tensorized = False
        
        logger = logging.getLogger(__name__)

        if tensorized:
            # check if tensorized model already exists so we can skip expensive model loading below
            _path, exists = tensorized_path(model_name)
            if exists:
                logger.info(f'Loading tensorized model {model_name}')
                self.model = untensorize(str(_path), self.device, quantized=quantized)
                self.tensorized = True

        elif (not quantized) and (not self.tensorized):
            self.model = AutoModelForSoftPromptLM.from_pretrained(model_name, return_dict_in_generate=True, torch_dtype=model_dtype).eval().to(self.device)

        if quantized:
            self.quantized = True
            logger.info(f'Quantizing model {model_name}')
            # we assume this is a gptj model - TODO: fix this
            transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock  # monkey-patch GPT-J
            if not self.tensorized:
                self.model = GPTJForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, return_dict_in_generate=True).eval().to(self.device)
            logger.info(f'Quantization complete.')
        else:
            self.quantized = False

        if (tensorized) and (not self.tensorized):
            # check if model file exists in ./storage/{model_name}.model
            _path, exists = tensorized_path(model_name)
            if not exists:
                logger.info(f'Tensorizing model {model_name}')
                # tensorize model
                tensorize(self.model, str(_path))
                del self.model
                raise Exception('Tensorized the model! The original model has been altered, please load the model again to use the tensorized model.')

        if parallelize:
            self.model.parallelize()

    @torch.inference_mode()  # inject deepspeed inference runtime where possible
    def generate(self, args: GPTGenRequest, *, db_softprompt = None) -> Response:  # TODO: refactor, use new sfmodel optional tag later
        logits_warpers = []
        logits_processors = []
        stopping_criterion = []
        eos_token_id = args.gen_args.eos_token_id if args.gen_args.eos_token_id else None
        softprompt = None
        output_scores = (args.gen_args.logprobs or args.gen_args.best_of)
        best_of = args.gen_args.best_of if args.gen_args.best_of else None
        prompt_length = None

        if db_softprompt:  # move this elsewhere?
            tbuf = np.frombuffer(zlib.decompress(db_softprompt.read()), dtype=np.float16)
            tensor = torch.from_numpy(np.array(tbuf).reshape(20, len(tbuf)//20)).to(self.device)
            softprompt = SoftPrompt(softembedding=tensor)
            sp_ids = [[id] for id in softprompt.get_special_ids()]
            logits_processors.append(NoBadWordsLogitsProcessor(sp_ids, None))

        # Stopping criteria
        prompt_length = len(self.tokenizer.encode(args.prompt))
        if softprompt:
            prompt_length += 20
        stopping_criterion.append(MaxLengthCriteria(args.gen_args.max_length + prompt_length))

        if args.gen_args.max_time: stopping_criterion.append(MaxTimeCriteria(args.gen_args.max_time))
        if args.gen_args.min_length: logits_processors.append(MinLengthLogitsProcessor(args.gen_args.min_length, eos_token_id))

        # Warpers
        if args.sample_args.temp: logits_warpers.append(TemperatureLogitsWarper(args.sample_args.temp))
        if args.sample_args.top_p: logits_warpers.append(TopPLogitsWarper(args.sample_args.top_p))
        if args.sample_args.top_k: logits_warpers.append(TopKLogitsWarper(args.sample_args.top_k))
        if args.sample_args.top_a: logits_warpers.append(TopALogitsWarper(args.sample_args.top_a))
        if args.sample_args.typical_p: logits_warpers.append(TypicalLogitsWarper(args.sample_args.typical_p))
        if args.sample_args.tfs: logits_warpers.append(TailFreeSamplingLogitsWarper(args.sample_args.tfs))

        # Processors
        if args.sample_args.rep_p:
            logits_processors.append(RepetitionPenaltyLogitsProcessor(penalty=args.sample_args.rep_p,
            slope=args.sample_args.rep_p_slope,
            penalize_last=args.sample_args.rep_p_range))
        if bad_words := args.sample_args.bad_words:
            bad_words_ids = []
            for bad_word in bad_words:
                bad_words_ids.append(self.tokenizer.encode(bad_word))
            logits_processors.append(NoBadWordsLogitsProcessor(bad_words_ids, None))
        if logit_biases := args.sample_args.logit_biases:
            _logit_biases = []
            for logit_bias in logit_biases:
                _logit_biases.append((logit_bias.id, logit_bias.bias))
            logits_processors.append(LogitBiasProcessor(_logit_biases))
        if phrase_biases := args.sample_args.phrase_biases:
            for phrase_bias in phrase_biases:
                logits_processors.append(PhraseBiasProcessor([self.tokenizer.encode(sequence) for sequence in phrase_bias.sequences], phrase_bias.bias, phrase_bias.ensure_sequence_finish, phrase_bias.generate_once))

        logits_warper = LogitsProcessorList(logits_warpers)
        logits_processor = LogitsProcessorList(logits_processors)
        stopping_criteria = StoppingCriteriaList(stopping_criterion)

        # Generate
        output = Response()
        best_of_idx = 0

        global current_sp
        current_sp = softprompt
        if softprompt:
            sp_tokenizer = softprompt.get_tokenizer(self.tokenizer)
            resize_model_embeddings(self.model, sp_tokenizer)
            input_ids = sp_tokenizer(args.prompt, return_tensors='pt').to(self.device)
        else:
            resize_model_embeddings(self.model, self.tokenizer)
            input_ids = self.tokenizer(args.prompt, return_tensors='pt').to(self.device)
        
        outputs = None
        if best_of is None:
            outputs = self.model.sample(
                **input_ids,
                logits_warper=logits_warper,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores
            )
        else:
            best_of_outputs = []
            best_of_sequences = []
            for i in range(best_of):
                outputs = self.model.sample(
                    **input_ids,
                    logits_warper=logits_warper,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=eos_token_id,
                    output_scores=output_scores
                )
                scores = []
                for token_idx in range(len(outputs.sequences[0]) - prompt_length):
                    scores.append(outputs.scores[token_idx][0][outputs.sequences[0][token_idx + prompt_length]].detach().item())
                best_of_sequences.append(torch.tensor(scores).mean().detach().item())
                best_of_outputs.append(outputs)
            best_of_idx = best_of_sequences.index(max(best_of_sequences))
            outputs = best_of_outputs[best_of_idx]

        output.output = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
#        if softprompt:
#            output["output"] = output["output"][len(softprompt.get_special_str()):]
            
        if logprobs := args.gen_args.logprobs:
            _logprobs = []
            for i in range(len(outputs.scores)):
                logprobs_seq = []
                scores_probs = outputs.scores[i].softmax(-1).topk(logprobs, dim=-1).values.tolist()
                scores_indices = outputs.scores[i].topk(logprobs, dim=-1).indices.tolist()
                for j in range(logprobs):
                    logprobs_seq.append((scores_indices[0][j], scores_probs[0][j]))
                _logprobs.append(logprobs_seq)
            output.logprobs = _logprobs
        
        return output

    @torch.inference_mode()
    def classify(self, prompt: str, labels: list[int]):

        prompt_inputs = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)

        output_probs = {}
        for i in labels:
            label_inputs = self.tokenizer(i, return_tensors='pt').input_ids.to(self.device)
            probs = self.model.forward(input_ids=torch.cat([prompt_inputs, label_inputs], dim=-1)).logits.softmax(-1)[0][-len(label_inputs[0]):]
            token_probs = [probs[t][label_inputs[0][t]] for t in range(0, len(label_inputs[0]))]
            output_probs[i] = torch.mean(torch.stack(token_probs, dim=-1)).item() 

        return output_probs
    
    @torch.inference_mode()
    def hidden(self, prompt: str, layers: List[int]):
        """
        Return hidden states from a forward pass
        :param prompt: prompt to extract hidden states from
        :param layers: number of last hidden layers to return
        """
        
        prompt_inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        hidden_states = self.model(prompt_inputs, output_hidden_states=True).hidden_states
        layers = {i: torch.mean(hidden_states[i], dim = (1, )).detach().cpu().numpy().tolist() for i in layers}
        
        return layers
