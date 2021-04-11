# KEnS
Resources and code for paper "Multiplingual Knowledge Graph Completion via Ensemble Knowledge Transfer"


## Install
Make sure your local environment has the following installed:

    python==3.6
    pandas
    tensorflow==1.10.0
    numpy==1.16.2
    pandas==1.0.3
    
Install the dependents using:

    pip install -r requirements.txt

## Run the experiments
1. To train the model for KEnS(RotatE), use:

    python ./run.py --knowledge_model rotate --target_language ja --use_default
    
    
You can use `-m transe` to switch to the KEnS(TransE) and change `--target_language` to `es, el, en, fr` to train for different target languages. `--use_default` means to use the preset hyper-parameters. By default, the trained models are saved in `$PROJECT_DIR$/trained_model/kens-$KNOWLEDGE_MODEL$-$DIM$/$TARGET_LANGUAGE$`.

To set your own hyper-parameters:

    python ./run.py --knowledge_model rotate --target_language ja -d 400 -b 2048 -lr 1e-2 --rotate_gamma 24 --reg_scale 1e-4 --base_align_step 5 --knowledge_step_ratio 20 --align_lr 1e-3


2. To test the trained model with the ensemble technique, use:

    python ./test.py --knowledge_model rotate --target_language ja --model_dir $TRAINED_MODEL_DIR$  -d $YOUR_MODEL_DIM$
    


## Reference
Please refer to our paper:

Xuelu Chen, Muhao Chen, Changjun Fan, Ankith Uppunda, Yizhou Sun, Carlo Zaniolo. Multilingual Knowledge Graph Completion via Ensemble Knowledge T. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings*, 2020

    @inproceedings{chen2020multilingual,
      title={Multilingual Knowledge Graph Completion via Ensemble Knowledge Transfer},
      author={Chen, Xuelu and Chen, Muhao and Fan, Changjun and Uppunda, Ankith and Sun, Yizhou and Zaniolo, Carlo},
      booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings},
      pages={3227--3238},
      year={2020}
    }
