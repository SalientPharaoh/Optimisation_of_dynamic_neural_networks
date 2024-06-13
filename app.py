from flask import Flask, request, jsonify,send_from_directory
import wandb
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset
from helper_util import *
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

@app.route('/api/base', methods=['POST'])
def base_model():
    data = request.json
    model_name = data.get('model_name')
    dataset_name = data.get('dataset_name')
    huggingface_token = data.get('huggingface_token')
    wandb_token = data.get('wandb_token')
    model_subset = data.get('model_subset')
    train_arg = data.get('train_arg')
    wandb.login(key=wandb_token)
    run = wandb.init(project="D2NN-flask", name="baseline")
    run_id = run.id

    eval_results = base_inference(model_name, dataset_name, wandb_token, huggingface_token, train_arg)

    wandb.log(eval_results)
    wandb.finish()

    return jsonify({'eval_results':eval_results,'run_id': run_id})



@app.route('/api/optimise', methods=['POST'])
def optimise():
    data = request.json
    model_name = data.get('model_name')
    dataset_name = data.get('dataset_name')
    huggingface_token = data.get('huggingface_token')
    wandb_token = data.get('wandb_token')
    model_subset = data.get('model_subset')
    train_arg = data.get('train_arg')

    wandb.login(key=wandb_token)
    run = wandb.init(project="D2NN-flask", name="optimised")
    run_id = run.id

    eval_results = opt_inference(model_name, dataset_name, wandb_token, huggingface_token, train_arg)

    wandb.log(eval_results)
    wandb.finish()

    return jsonify({'eval_results':eval_results,'run_id': run_id})

@app.route('/api/zeroquant', methods=['POST'])
def optimise_zeroquant():
    data = request.json
    model_name = data.get('model_name')
    dataset_name = data.get('dataset_name')
    huggingface_token = data.get('huggingface_token')
    wandb_token = data.get('wandb_token')
    model_subset = data.get('model_subset')
    train_arg = data.get('train_arg')

    wandb.login(key=wandb_token)
    run = wandb.init(project="D2NN-flask", name="Zero_Quant")
    run_id = run.id

    eval_results = zeroQuant(model_name, dataset_name, wandb_token, huggingface_token, train_arg)

    wandb.log(eval_results)
    wandb.finish()
    return jsonify({'eval_results':eval_results,'run_id': run_id})

@app.route('/api/XTC', methods=['POST'])
def optimise_XTC():
    data = request.json
    model_name = data.get('model_name')
    dataset_name = data.get('dataset_name')
    huggingface_token = data.get('huggingface_token')
    wandb_token = data.get('wandb_token')
    model_subset = data.get('model_subset')
    train_arg = data.get('train_arg')

    wandb.login(key=wandb_token)
    run = wandb.init(project="D2NN-flask", name="Extreme_Compression")
    run_id = run.id

    eval_results = XTC(model_name, dataset_name, wandb_token, huggingface_token, train_arg)
    wandb.log(eval_results)
    wandb.finish()
    return jsonify({'eval_results':eval_results,'run_id': run_id})

@app.route('/api/weight_quant', methods=['POST'])
def optimise_WQ():
    data = request.json
    model_name = data.get('model_name')
    dataset_name = data.get('dataset_name')
    huggingface_token = data.get('huggingface_token')
    wandb_token = data.get('wandb_token')
    model_subset = data.get('model_subset')
    train_arg = data.get('train_arg')

    wandb.login(key=wandb_token)
    run = wandb.init(project="D2NN-flask", name="Weight_Quantization")
    run_id = run.id

    eval_results = weight_Quant(model_name, dataset_name, wandb_token, huggingface_token, train_arg)
    wandb.log(eval_results)
    wandb.finish()
    return jsonify({'eval_results':eval_results,'run_id': run_id})

@app.route('/api/pruning', methods=['POST'])
def optimise_prune():
    data = request.json
    model_name = data.get('model_name')
    dataset_name = data.get('dataset_name')
    huggingface_token = data.get('huggingface_token')
    wandb_token = data.get('wandb_token')
    model_subset = data.get('model_subset')
    train_arg = data.get('train_arg')

    wandb.login(key=wandb_token)
    run = wandb.init(project="D2NN-flask", name="Unstructured_Pruning")
    run_id = run.id

    eval_results = pruning(model_name, dataset_name, wandb_token, huggingface_token, train_arg)
    wandb.log(eval_results)
    wandb.finish()
    return jsonify({'eval_results':eval_results,'run_id': run_id})

@app.route('/api/PTQuant', methods=['POST'])
def optimise_PTQ():
    data = request.json
    model_name = data.get('model_name')
    dataset_name = data.get('dataset_name')
    huggingface_token = data.get('huggingface_token')
    wandb_token = data.get('wandb_token')
    model_subset = data.get('model_subset')
    train_arg = data.get('train_arg')

    wandb.login(key=wandb_token)
    run = wandb.init(project="D2NN-flask", name="Post_Training_Quantization")
    run_id = run.id

    eval_results = PTQuant(model_name, dataset_name, wandb_token, huggingface_token, train_arg)
    wandb.log(eval_results)
    wandb.finish()
    return jsonify({'eval_results':eval_results,'run_id': run_id})

@app.route('/api/w8a8', methods=['POST'])
def optimise_w8a8():
    data = request.json
    model_name = data.get('model_name')
    dataset_name = data.get('dataset_name')
    huggingface_token = data.get('huggingface_token')
    wandb_token = data.get('wandb_token')
    model_subset = data.get('model_subset')
    train_arg = data.get('train_arg')

    wandb.login(key=wandb_token)
    run = wandb.init(project="D2NN-flask", name="W8A8_Quantization")
    run_id = run.id

    eval_results = w8a8_Quant(model_name, dataset_name, wandb_token, huggingface_token, train_arg)
    wandb.log(eval_results)
    wandb.finish()
    return jsonify({'eval_results':eval_results,'run_id': run_id})



@app.route('/api/download/<path:filename>', methods=['GET'])
def download_file(filename):
    path =  os.getcwd()
    directory = os.path.abspath(path + '/OptimisedModel')
    return send_from_directory(directory, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)