from transformers import Trainer, TrainerArguments, DataCollatorForVLACausalLM
from sklearn.model_selection import train_test_split
import pandas as pd

MODEL_PATHS = {'Med3DVLM': "/mnt/fac/CX000019_DS1/UCSF-PGDM/PKG_-_UCSF-PDGM_Version_5/UCSF-PDGM-v5",
            'MedGemma' : ""}

def qa_metrics(eval_pred):
    """
    This function is used to show how the prediction should be compared to the result.
    """
    #Get the 
    predictions, labels = eval_pred

    correct = 0
    for idx in range(len(predictions)):
        if label[idx] in predictions[idx]:
            correct += 1

    return correct / len(predictions)


def finetune (model_name: str, dataset):
    """
    The finetune function is used to properly train each model based on their
    own parts. For medgemma, we need a processor and the model. For Med3DVLM,
    it has a model and tokenizer.

    model_parts = (model, other parts (processor, tokenizer, etc.))
    dataset = QApairs used to fine tune (Should be a Dataframe)
    """

    #Split the dataset into train, validate, test 
    qa_train, qa_eval = train_test_split(dataset, train_size=0.7)
    qa_val, qa_test = train_test_split(qa_eval, train_size=0.5)


    #Initialize the training arguments for the Trainer object
    training_args = TrainingArguments(
        output_dir="brain_qa_evaluater",
        eval_strategy="epoch",
        push_to_hub=False,
    )

    if model_name == 'Med3DVLM':
        #Place all the related code for finetuning Med3DVLM
        #Set up the model
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATHS[model],
            model_max_length=512,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATHS[model],
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        #Further set up the dataset so model can take the images
        proj_out_num = getattr(model.get_model().config, "proj_out_num", 256)
        train_ds = BrainVLMDataset(
            examples=qa_train,
            tokenizer=tokenizer,
            image_dir="/mnt/fac/CX000019_DS1/UCSF-PGDM/PKG_-_UCSF-PDGM_Version_5/UCSF-PDGM-v5",
            proj_out_num=proj_out_num,
            target_size=(128, 256, 256)
        )

        collator = DataCollatorForVLACausalLM(tokenizer, pad_to_multiple_of=8)

        #Set up the trainer object
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=qa_train,
            eval_dataset=qa_val,
            data_collator=collator,
            compute_metrics=qa_metrics,
        )
        trainer.train()


    elif model_name == 'medgemma-1.5-4b-it'
        #Place all the related code for finetuning Medgemma ()


def main():
    model='Med3DVLM'
    data = pd.read_csv("/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/UCSF_PDGM_QAPairs_Sample.csv")
    
    finetune(model, data)

if __name__ == "__main__":
    main()