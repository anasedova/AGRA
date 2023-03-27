def define_model_name(model_type, dataset):
    if dataset in ['yoruba', 'hausa'] and model_type == "bert":
        return 'bert-base-multilingual-cased'
    elif dataset in ['youtube', 'sms', 'trec', 'yoruba', 'hausa'] and model_type == "bert":
        return 'roberta-base'
    elif dataset in ['cifar', 'chexpert'] and model_type == "resnet":
        return 'resnet50'
    else:
        raise ValueError(f"Are you sure you want to use the combination of {dataset} dataset and {model_type} model?")

