params = {'model': {'input_size': 384,
                    'hidden_size': 2048,
                    'num_lstm': 2,
                    'bidirectional': True,
                    'batch_first': True,
                    'num_linear': 3,
                    'num_classes': 4,
                    'dropout': 0.4},
          'train': {'epochs': 100,
                    'optimizer': 'Adam',
                    'learning_rate': 1e-4,
                    'weight_decay': 1e-5,
                    'momentum': 0.9,
                    'device': 'cuda'},
          'data': {'batch_size': 4,
                   'shuffle': True}
          }

