import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from utils.scraper import HFModelScraper
from models.model_handler import ModelHandler

if __name__=="__main__":
    scraper = HFModelScraper("bert-base-uncased")
    model = ModelHandler("bert-base-uncased")
    model_class, tokenizer = scraper.get_model_classes()
    model.load_model(model_class, tokenizer)
    print(model.model)
    print(model.tokenizer)