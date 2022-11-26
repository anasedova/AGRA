import torch

from wrench.endmodel.bert_model_with_AGRA import BertClassifierModelWithAGRA
from wrench.dataset import load_dataset
from wrench.labelmodel.majority_voting import MajorityVoting
from src.single_label.utils import set_seed
from snorkel.utils import probs_to_preds


if __name__ == '__main__':
    dataset_path = r'C:\Users\Lena\Documents\PhD - Preparation\GM Paper\New Code\datasets\datasets'
    data = 'youtube'
    extractor = 'tfidf'
    input_data_path = r"C:\Users\Lena\Documents\PhD - Preparation\GM Paper\New Code\datasets\datasets"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = 'roberta-base'

    set_seed(0)

    train_data, valid_data, test_data = load_dataset(
        dataset_path,
        data,
        extract_feature=False
        )

    # get the labels by majority voting for train data
    label_model = MajorityVoting()
    label_model.fit(dataset_train=train_data)
    soft_labels_train = label_model.predict_proba(train_data)
    y_train = probs_to_preds(soft_labels_train)

    model = BertClassifierModelWithAGRA(
            model_name=model_name,
            batch_size=3,
            comp_batch_size=10,
            test_batch_size=3,
            comp_loss="CE",
            other=0,
            agra_threshold=0,
            n_steps=10
        )

    model.fit(
        dataset_train=train_data,
        y_train=y_train,
        dataset_valid=valid_data,
        evaluation_step=2,
        metric='acc',
        patience=100,
        device=device
    )
    acc = model.test(test_data, 'acc')