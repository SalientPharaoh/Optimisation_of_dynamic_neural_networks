import importlib
import torch
import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from utils.scraper import HFModelScraper

def load_class(module_name, class_name):
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls

class ModelHandler:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_datacard = None
        self.model_class, self.tokenizer_class = self.get_model_details()

    def get_model_details(self):
        scraper = HFModelScraper(self.model_name)
        self.model_datacard = scraper.get_model_information()
        return scraper.get_model_classes()


    def load_model(self):
        auto_model_class_name = self.model_class
        auto_tokenizer_class_name = self.tokenizer_class
        
        AutoModelClass = load_class('transformers', auto_model_class_name)
        AutoTokenizerClass = load_class('transformers', auto_tokenizer_class_name)

        self.tokenizer = AutoTokenizerClass.from_pretrained(self.model_name)
        self.model = AutoModelClass.from_pretrained(self.model_name).to(self.device)

    def run_inference(self, inputs):
        inputs = self.tokenizer(inputs, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs

