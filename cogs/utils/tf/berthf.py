from typing import List
import torch
import ray
from ray import serve

import logging

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

from .autohf import AutoHF
from .tensorize import tensorize, untensorize
from .utils import get_dtype, tensorized_path

@serve.deployment(num_replicas=2, name="BERT", ray_actor_options={"num_gpus": 1})
class BERTHF(AutoHF):
    def __init__(self, model_name='distilroberta-base', device=None, parallelize=False, quantized=False, tensorized=False):
        super().__init__(model_name=model_name, decoder=False)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        model_dtype = get_dtype(device)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tensorized = False

        logger = logging.getLogger(__name__)

        if tensorized:
            _path, exists = tensorized_path(model_name)
            if exists:
                logger.info(f'Loading tensorized model {model_name}')
                self.model = untensorize(str(_path), self.device, quantized=quantized)
                self.tensorized = True
        
        if (not quantized) and (not self.tensorized):
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=model_dtype
            ).eval().to(self.device)
        
        if quantized:
            raise NotImplementedError('Quantized models are not supported yet for encoder models such as BERT.')
        
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
            raise NotImplementedError('Parallelization is not supported yet for encoder models such as BERT.')
    
    @torch.inference_mode()
    def classify(self, prompt: str, labels: List[int]):
        
        prompt_inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        outputs = self.model(prompt_inputs).logits
        outputs = torch.nn.functional.softmax(outputs, dim=1)
        outputs = outputs.detach().cpu().numpy()
        output_probs = {}

        # TODO: automatically fill labels

        for i in range(len(labels)):
            output_probs[labels[i]] = float(outputs[0][i])

        return output_probs

    @torch.inference_mode()
    def hidden(self, prompt: str, layers: List[int]):
        # args:
        #   prompt: str - prompt to extract hidden states from
        #   layers: int - number of last hidden layers to return
        
        prompt_inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        hidden_states = self.model(prompt_inputs, output_hidden_states=True).hidden_states
        layers = {i: torch.mean(hidden_states[i], dim = (1, )).detach().cpu().numpy().tolist() for i in layers}
        
        return layers
