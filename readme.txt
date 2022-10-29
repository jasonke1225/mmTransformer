conda name: mmTrans
1. python 3.8(3.7會因為preprocessed data是3.8而不行), cuda 11.4
=> conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

2. using AdamW
=> conda install -c esri tensorflow-addons
=> import tensorflow as tf
   import tensorflow_addons as tfa
=> optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
   torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1) # gradient max norm is 0.1
   optimizer.step()
   optimizer.zero_grad(set_to_none=True)

3. element 0 of tensors does not require grad and does not have a grad_fn
=> loss.requires_grad_(True)

4. Cuda out of memory
=> batch_size 改 64

5. local variable 'beta1' referenced before assignment
=> 用pytorch 1.9以上的

6. GPU使用率低
=> 調整dataloader的num_worker為CPU核心數(8)

7. 為了加速loss function，盡可能轉成tensor去作為input和target

8. training 到非第一個epoch過程 cuda out of memory
=> del loss_score (將train函式內用不到的變數都del)
   gc.collect()
   torch.cuda.empty_cache()
=> optimizer.zero_grad(set_to_none=True)
=> 減少batch size 和 gradient accumulation

9. 計算Loss時，pred為(batch,6,30,2)，要用permute(1,0,2,3)去變成(6,batch,30,2)，用view會錯，詳情可看loss.py

10. loss一直沒變化
=> 不使用FormatData()
=> 因為它使用了detach和將tensor轉np，破壞了tensor內存的參數大小

11. received 0 items of ancdata 
=> torch.multiprocessing.set_sharing_strategy('file_system')
