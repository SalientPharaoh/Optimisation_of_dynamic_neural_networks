import torch
from torch.utils.data import DataLoader
import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from models.model_handler import ModelHandler

def main():
    logger = get_logger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Loading the Model and Tokenizer
    model_name = "bert-base-uncased"
    modelHandler = ModelHandler(model_name)
    modelHandler.load_model()

    #Loading the test_dataset
    dataset_name = "glue"
    subset = "mrpc"
    dataHandler = DatasetLoader(modelHandler.tokenizer, dataset_name, subset)
    dataloader = dataHandler.get_dataloader(batch_size=8)
    
    modelInference = ModelMetrics(modelHandler.model, dataloader, device)
    modelInference.calculate_accuracy()    

if __name__ == "__main__":
    main()
