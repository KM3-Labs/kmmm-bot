from argparse import ArgumentError
from multiprocessing.sharedctypes import Value
from optparse import Option
from typing import Optional
from shimeji.model_provider import ModelProvider
import logging

import ray
from ray import serve


# Everything written here MUST not be dependent on any non-inference related code!
class ServalModelProvider(ModelProvider):
    def __init__(self, address="auto", gpt_model: Optional[str] = None, bert_model: Optional[str] = None, fid_model: Optional[str] = None):
        ray.init(address, namespace="serval")

        # these are just guidelines; absolutely make sure you're assigning the right models... i made it painfully obvious
        self.gpt_d = None
        self.bert_d = None
        self.fid_d = None

        # check deployments and see if they match up
        for d_name, d in serve.list_deployments():
            if gpt_model == d_name: self.gpt_d = d
            if bert_model == d_name: self.bert_d = d
        
        if (self.gpt_d is not None) and (self.bert_d is not None):
            logging.info(f"Using {gpt_model} for language generation and {bert_model} for embeddings tasks.")
            # load model if not available
            
        elif (self.gpt_d is not None) and (self.bert_d is None):
            logging.info(f"Using {gpt_model} for generative and embeddings tasks.")  # add option to opt out of embeddings stuff
        elif (self.gpt_d is None) and (self.bert_d is not None):
            logging.info(f"Using {bert_model} for embeddings tasks.")
            logging.warn("No causal model loaded. Disabled generative conversations.")
        else:  # Nothing got loaded.
            raise ValueError(f"No valid GPT or BERT model names provided! The available deployments are:\n{serve.list_deployments()}")
        
        # handles here? if no bert, use gpt hidden states. make use of Deployment.options()
    
    

    def auth(self):
        pass  # cloud or remote cluter related stuff here, put all local stuff in constructor including docker api usage


    async def generate(self, args):
        if self.gpt_d is None:
            raise ValueError("No causal model loaded")
        pass


    async def get_hidden_states(self, args):
        pass

    async def should_respond(self, context, name):
        pass
