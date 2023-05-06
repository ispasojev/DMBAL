from playground.dmbal_centroids import train
# Input set by user to run this script:

# Hyperparameter:
# - beta
# - type of k (number of classes vs. batch size)
# - type_of_k/k_strategy (values=’n_classes’ / ‘batch_size’)
# - k_equals_batch_size (True / False)
# - k (type int)
# - dataset (e.g. cifar10)
# - model (e.g. resnet34)
# - train_batch_size
# - sampling_batch_size
# - budget
# - sampling_strategy (e.g. ‘original’ / ‘boundary’)
# - random_seed
# - max_epoch

if __name__ == '__main__':
    train(log=False)