import UtilsCifar10
from deepstack.base import KerasMember
from deepstack.ensemble import DirichletEnsemble
from deepstack.ensemble import StackEnsemble
from sklearn.ensemble import RandomForestRegressor


def cifar10_example(nmembers=4):
    """
    Runs 2 DeepStack Ensemble Models for the CIFAR-10 Dataset.

    Args:
        nmembers: amount of ensemble members to be generated

    Returns: an instance of StackEnsemble and DirichletEnsemble for the
    CIFAR-10 dataset
    """
    stack = StackEnsemble()  # Meta-Learner
    stack.model = RandomForestRegressor(verbose=1, n_estimators=300 * nmembers,
                                        max_depth=nmembers * 2, n_jobs=4)
    dirichletEnsemble = DirichletEnsemble(N=2000 * nmembers)

    for i in range(nmembers):
        # Creates a Random CNN Keras Model for CIFAR-10 Dataset
        model, training_batch, validation_batch = UtilsCifar10.get_random_cifar_model()
        """
        Rationale: The Validation and Testing dataset of a base-learner
        is the Training and Validation Dataset of a Meta-Learner.
        Idea is to avoid validating the meta-learner
        on data that the base-learner has already seen on training
        """
        member = KerasMember(name="model" + str(i + 1), keras_model=model,
                             train_batches=training_batch,
                             val_batches=validation_batch)  # Base-Learners
        stack.add_member(member)  # Adds base-learner to Stack Ensemble
        dirichletEnsemble.add_member(member)  # Adds base-learner
    stack.fit()
    dirichletEnsemble.fit()
    return stack, dirichletEnsemble


if __name__ == "__main__":
    # Run Examples
    stack, dirichletEnsemble = cifar10_example()

    stack.describe()
    """
    Possible similar Output:
    model1 - AUC: 0.8044
    model2 - AUC: 0.8439
    model3 - AUC: 0.8218
    model4 - AUC: 0.8487
    StackEnsemble AUC: 0.8727
    """

    dirichletEnsemble.describe()
    """
    Possible similar Output:
    model1 - Weight: 0.1055 - AUC: 0.8044
    model2 - Weight: 0.2882 - AUC: 0.8439
    model3 - Weight: 0.2127 - AUC: 0.8218
    model4 - Weight: 0.3936 - AUC: 0.8487
    DirichletEnsemble AUC: 0.8821
    """
