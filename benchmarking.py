import graphgym
from graphgym.config import cfg
from graphgym.register import register_data_args, register_model_args, register_trainer_args
from graphgym.loader import DataLoader
from graphgym.model_builder import create_model
from graphgym.utils import set_seed

def main():
    # Set the random seed
    set_seed(cfg.seed)

    # Load the dataset
    data = DataLoader(cfg)

    # Create the model
    model = create_model(cfg)

    # Train and evaluate the model
    trainer = graphgym.trainer.Trainer(cfg, model, data)
    trainer.train()

if __name__ == '__main__':
    # Set the file path of your custom graph dataset
    cfg.data.dataset = '/path/to/your/dataset'

    # Set the models to run
    cfg.model.name = ['pytorch_model', 'pytorch_geometric_model']

    # Set the tasks to test the models on
    cfg.task.name = ['task1', 'task2', 'task3']

    # Set the number of epochs to run the models for
    cfg.train.epochs = 100

    # Parse the command line arguments
    register_data_args(cfg)
    register_model_args(cfg)
    register_trainer_args(cfg)
    cfg.merge_from_file('config.yaml')  # Optional: Load additional configuration from a YAML file

    # Run the script
    main()
