# noun phrases model training

## Overview
A PyTorch based implementation of model for context based noun similarity

## Running the code
Run main_task.py with arguments below. require generating tripplets from the amazon review corpus for train and test.
```bash
main_pretraining.py [-h] 
	[--pretrain_data TRAIN_PATH] 
	[--batch_size BATCH_SIZE]
	[--freeze FREEZE]  
	[--gradient_acc_steps GRADIENT_ACC_STEPS]
	[--max_norm MAX_NORM]
	[--fp16 FP_16]  
	[--num_epochs NUM_EPOCHS]
	[--lr LR]
```
