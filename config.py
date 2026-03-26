class Config():
    """Configuration for a contrastive learning run."""
    def __init__(self,
                 # Fingerprint configuration
                 fp_algo,
                 fp_type,
                 fp_length,
                 fp_radius,
                 
                 # Contrastive learning
                 emb_dim,
                 # Fingerprint encoder
                 fp_hidden_dim,
                 # Isotopic pattern encoder
                 patt_hidden_dim1,
                 patt_hidden_dim2,
                 # Loss temperature
                 tau_clip,
                 
                 # Training
                 batch_size,
                 seed,
                 
                 # arguments with default values
                 fp_dropout=0,
                 patt_dropout=0,
                 learning_rate=1e-4,
                 patience=1000,
                 ema_alpha=0.1,
                 max_grad_norm=1.0,
                 max_steps=int(1e9),
                 max_epochs=int(1e9),
                 log_every_epochs=1,
                 log_every_steps=50,
                 label_smoothing_clip=0.0,
                 learnable_temperature=False
                 ):
        self.config_dict = {
            'fp_algo': fp_algo,
            'fp_type': fp_type,
            'fp_length': fp_length,
            'fp_radius': fp_radius,
            
            'emb_dim': emb_dim,
            'tau_clip': tau_clip,
            'label_smoothing_clip': label_smoothing_clip,
            'learnable_temperature': learnable_temperature,
            
            'fp_hidden_dim': fp_hidden_dim,
            'fp_dropout': fp_dropout,
            
            'patt_hidden_dim1': patt_hidden_dim1,
            'patt_hidden_dim2': patt_hidden_dim2,
            'patt_dropout': patt_dropout,
            
            'batch_size': batch_size,
            'seed': seed,
            'learning_rate': learning_rate,
            'patience': patience,
            'ema_alpha': ema_alpha,
            'max_grad_norm': max_grad_norm,
            'max_steps': max_steps,
            'max_epochs': max_epochs,
            'log_every_epochs': log_every_epochs,
            'log_every_steps': log_every_steps
        }

    def get_param(self, param_name):
        return self.config_dict[param_name]
