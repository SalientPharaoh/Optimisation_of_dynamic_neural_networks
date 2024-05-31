import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from utils.scraper import HFModelScraper

def main(model_name):
    scraper = HFModelScraper(model_name)
    tags, library_name, config, transformers_info = scraper.get_model_information()
    model_class, tokenizer_class = scraper.get_model_classes()
    print("Model Class:", model_class)
    print("Tokenizer Class:", tokenizer_class)

if __name__ == "__main__":
    model_name = "bert-base-uncased"
    main(model_name)