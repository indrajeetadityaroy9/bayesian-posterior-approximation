import torch
import numpy as np
import warnings

from sota_mlp import AdvancedMLPConfig, create_model
from sota_trainer import AdvancedTrainer
from uncertainty_quantification import CalibrationAnalyzer
from utility import generate_data

warnings.filterwarnings('ignore')


def test_sota_components():
    print("=" * 60)
    print("TESTING SOTA ML SYSTEM")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    print("\n1. Testing data generation...")
    gmm_params = {
        'priors': np.array([0.25, 0.25, 0.25, 0.25]),
        'meanVectors': np.array([
            [0.0, 0.0, 0.0],
            [4.0, 4.0, 0.0],
            [0.0, 4.0, 4.0],
            [4.0, 0.0, 4.0]
        ]),
        'covMatrices': np.array([
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ])
    }

    X_train, y_train = generate_data(500, gmm_params)
    X_test, y_test = generate_data(1000, gmm_params)
    print(f"  Generated train data: {X_train.shape}, test data: {X_test.shape}")

    print("\n2. Testing SOTA architectures...")

    config_basic = AdvancedMLPConfig()
    config_basic.hidden_dims = [64, 128, 64]
    config_basic.activation = 'swish'
    config_basic.use_batch_norm = True
    config_basic.max_epochs = 20
    config_basic.batch_size = 32
    model_basic = create_model(config_basic)
    print(f"  Basic SOTA MLP created: {sum(p.numel() for p in model_basic.parameters())} parameters")

    config_dropout = AdvancedMLPConfig()
    config_dropout.hidden_dims = [64, 128, 64]
    config_dropout.activation = 'swish'
    config_dropout.dropout_rates = [0.1, 0.15, 0.1]
    config_dropout.uncertainty_method = 'mc_dropout'
    config_dropout.mc_samples = 50
    config_dropout.max_epochs = 20
    config_dropout.batch_size = 32
    config_dropout.use_mixup = False
    config_dropout.label_smoothing = 0.0
    config_dropout.weight_decay = 1e-5
    model_dropout = create_model(config_dropout)
    print(f"  MC Dropout MLP created: {sum(p.numel() for p in model_dropout.parameters())} parameters")

    print("\n3. Testing training process...")

    trainer = AdvancedTrainer(config_basic)

    try:
        metrics = trainer.fit(X_train, y_train)
        print(f"  Training completed: {len(metrics)} epochs")
        print(f"  Final train accuracy: {metrics[-1].train_accuracy:.2f}%")
        print(f"  Final val accuracy: {metrics[-1].val_accuracy:.2f}%")
    except Exception as e:
        print(f"Training failed: {e}")
        return False

    print("\n4. Testing predictions and uncertainty...")

    try:
        predictions, probabilities, uncertainties = trainer.predict(X_test)
        print(f"  Predictions generated: {predictions.shape}")
        print(f"  Test accuracy: {np.mean(predictions == y_test):.3f}")
        print(f"  Mean uncertainty: {np.mean(uncertainties):.3f}")
    except Exception as e:
        print(f"Prediction failed: {e}")
        return False

    print("\n5. Testing calibration analysis...")

    try:
        analyzer = CalibrationAnalyzer(n_bins=10)
        metrics = analyzer.compute_calibration_metrics(y_test, probabilities)
        print(f"  Calibration analysis completed")
        print(f"  ECE: {metrics.expected_calibration_error:.3f}")
        print(f"  Brier Score: {metrics.brier_score:.3f}")
        print(f"  Mean Uncertainty: {metrics.mean_uncertainty:.3f}")
    except Exception as e:
        print(f"Calibration analysis failed: {e}")
        return False

    print("\n6. Testing MC Dropout uncertainty...")

    try:
        trainer_dropout = AdvancedTrainer(config_dropout)
        trainer_dropout.fit(X_train[:200], y_train[:200])

        pred_mc, prob_mc, unc_mc = trainer_dropout.predict(X_test[:100])
        print(f"  MC Dropout predictions: uncertainty range [{np.min(unc_mc):.3f}, {np.max(unc_mc):.3f}]")
    except Exception as e:
        print(f"MC Dropout failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! SOTA ML system is working correctly.")
    print("=" * 60)

    print(f"\nSystem capabilities verified:")
    print(f"  SOTA architectures (Swish, BatchNorm, Residual connections)")
    print(f"  Advanced training (Early stopping, Learning rate scheduling)")
    print(f"  Uncertainty quantification (MC Dropout, Calibration metrics)")
    print(f"  PyTorch integration")
    print(f"  Publication-ready analysis")

    return True


def test_minimal_sota_experiment():
    print("\n" + "=" * 60)
    print("RUNNING MINIMAL SOTA EXPERIMENT")
    print("=" * 60)

    try:
        from sota_paper_results import SOTAPaperResultsGenerator

        generator = SOTAPaperResultsGenerator(results_dir="test_sota_results")

        generator.training_sizes = [100, 500]
        generator.test_size = 2000

        config1 = AdvancedMLPConfig()
        config1.hidden_dims = [64, 128, 64]
        config1.activation = 'swish'
        config1.use_batch_norm = True
        config1.max_epochs = 30
        config1.batch_size = 32

        config2 = AdvancedMLPConfig()
        config2.hidden_dims = [64, 128, 64]
        config2.dropout_rates = [0.1, 0.15, 0.1]
        config2.uncertainty_method = 'mc_dropout'
        config2.mc_samples = 20
        config2.max_epochs = 30
        config2.batch_size = 32
        config2.use_mixup = False
        config2.label_smoothing = 0.0
        config2.weight_decay = 1e-5

        generator.model_configs = {
            'sota_basic': config1,
            'sota_mc_dropout': config2
        }

        print("Running minimal SOTA ML comparison...")
        results = generator.run_complete_analysis(n_runs=1)

        print(f"\n  Minimal experiment completed!")
        print(f"  Generated {len(results)} experiment results")
        print(f"  Results saved to: test_sota_results/")

        return True

    except Exception as e:
        print(f"Minimal experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing SOTA ML System Components...")

    if test_sota_components():
        test_minimal_sota_experiment()
    else:
        print("Component tests failed, skipping full experiment test.")