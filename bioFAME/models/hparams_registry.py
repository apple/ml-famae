# Record hyperparameters for the model

def _hparams(algorithm):
    """
    Global registry of hyperparams.
    """

    hparams = {}

    def _hparam(name, default_val, options=None):
        """Define a hyperparameter."""
        assert name not in hparams
        if options is not None:
            assert default_val in options
        hparams[name] = default_val

    def _update_hparam(name, new_val):
        assert name in hparams
        hparams[name] = new_val

        
    _hparam('patch_size', 20)
    _hparam('masking_ratio', 0.5)
    _hparam('encoder_only', False)
    _hparam('cls_token_mode', 'avg', ['avg', 'cls', 'special'])
    _hparam('dim', 64)
    _hparam('latent_expand', 'NA', ['NA', 'cat'])

    _hparam('trans1_depth', 4)
    _hparam('trans1_mlpdim', 128)
    _hparam('trans1_headdim', 16)

    _hparam('fft_head', 8)
    _hparam('fft_mode', 'pool', ['pool', 'proj', 'self'])

    _hparam('encoder_depth', 1)
    _hparam('modality-decoder', True)
    _hparam('post_norm', False)

    _hparam('time_embedding_token', False)
    _hparam('modality_embedding_token', False)

    _hparam('revin', True)
    _hparam('revin_recover', True)

    _hparam('include_Enc2', True)
    _hparam('mixup_mode', 'Mixup', ['NA', 'Mixup', 'CutMix'])

    return hparams