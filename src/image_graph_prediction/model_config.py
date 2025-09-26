class Config:
    # batch_size = 8
    # lr = 1e-4
    # epochs = 40
    # gradient_accumulation_steps = 1
    # weight_decay = 1e-2

    # for all data (saved)
    # batch_size = 32
    # lr = 1e-4
    # weight_decay = 1e-2
    # dropout = 0.2
    # gradient_accumulation_steps = 1
    # epochs = 18

    # batch_size = 16
    # lr = 3e-4
    # weight_decay = 1e-5
    # dropout = 0.5
    # gradient_accumulation_steps = 1
    # epochs = 100


    #for media eval
    batch_size = 32
    lr = 1e-5
    weight_decay = 1e-4     
    dropout = 0.2            
    epochs = 100        
    gradient_accumulation_steps = 1

  
    best_model_path = 'best_models/best_all_data_model.pth'
    # best_model_path = 'best_models/best_media_eval_model.pth'
    best_model_final_path = 'best_models/final_all_data_model_1.pth'
    # best_model_final_path = 'best_models/final_media_eval_model.pth'