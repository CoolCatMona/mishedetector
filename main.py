"""Determine if a face is Mishe or not."""
import logging
import os

from snn import SNN

logging.basicConfig(level=logging.INFO,
                    format='[%(name)s][%(asctime)s] %(levelname)s: %(message)s (%(filename)s:%(lineno)d)')
logger = logging.getLogger(__name__)


def main():
    input_directory = os.path.join(os.path.dirname(os.path.realpath(__name__)), 'anchors_positives')
    negatives = os.path.join(os.path.dirname(os.path.realpath(__name__)), 'negatives')
    evaluation = os.path.join(os.path.dirname(os.path.realpath(__name__)), 'evaluation')
    snn = SNN(
        dataset_path=input_directory,
        negatives_path=negatives,
        evaluation_path=evaluation,
    )
    # Final layer-wise momentum (mu_j in the paper)
    momentum = 0.9
    # linear epoch slope evolution
    momentum_slope = 0.01
    evaluate_each = 1000
    number_of_train_iterations = 10000
    validation_accuracy = snn.train_snn(number_of_iterations=number_of_train_iterations,
                                        final_momentum=momentum,
                                        momentum_slope=momentum_slope,
                                        evaluate_each=evaluate_each,
                                        model_name='siamese_net_lr10e-4')
    if validation_accuracy == 0:
        evaluation_accuracy = 0
    else:
        # Load the weights with best validation accuracy
        snn.model.load_weights('./models/siamese_net_lr10e-4.h5')
        evaluation_accuracy = snn.image_loader.one_shot_test(snn.model, number_of_tasks=40, is_validation=False)

    logger.info(f'Final Evaluation Accuracy = {evaluation_accuracy}')


if __name__ == "__main__":
    main()
