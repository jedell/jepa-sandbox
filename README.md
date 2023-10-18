
Pretrain:
    python main.py --unsupervised --data_dir=</path/to/data> --save_dir=<path/to/save/model>


Finetune:
    python main.py --finetune  --data_dir=</path/to/data> --save_dir=<path/to/save/model>

    --list_of_jepas: underscore dilineated list of the JEPA models with x number of frame skip to use in finetuning. ex: 0_5_10 for the 12th, 17th, and 22nd frame JEPAs




# text-jepa
