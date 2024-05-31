import requests
import json

class HFModelScraper:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_info = self.scrape_huggingface_info()
        

    def scrape_huggingface_info(self):
        api_url = f"https://huggingface.co/api/models/{self.model_name}"
        response = requests.get(api_url)
        if response.status_code == 200:
            model_info = response.json()
        else:
            model_info = {}
        return model_info

    def get_model_information(self):
        tags = self.model_info.get('tags', [])
        library_name = self.model_info.get('library_name', '')
        config = self.model_info.get('config', {})
        return (tags, library_name, config)

    def get_model_classes(self):
        try:
            transformers_info = self.model_info.get('transformersInfo', {})
            return transformers_info.get('auto_model', ''), transformers_info.get('processor', '')
        except Exception as e:
            print("Switching to default model and tokenizer classes")
            return 'AutoModel', 'AutoTokenizer'