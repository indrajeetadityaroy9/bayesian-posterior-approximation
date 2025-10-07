import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from bayesian_uq.benchmarks import GMMBenchmark
from bayesian_uq.evaluation import brier_score, compute_ece
from bayesian_uq.methods import get_method, list_methods, method_info


def create_benchmark(train_size=500, test_size=5000, seed=42):
    benchmark = GMMBenchmark(
        train_size=train_size, test_size=test_size, random_seed=seed
    )
    return (benchmark, benchmark.generate_data())


def get_method_config(method_name, args):
    config = {
        "hidden_dims": [128, 256, 128],
        "max_epochs": args.epochs,
        "batch_size": 64,
        "learning_rate": 0.001,
        "patience": 50,
        "verbose": False,
    }
    if method_name == "mc_dropout":
        config["mc_samples"] = 100
    elif method_name == "deep_ensemble":
        config["num_models"] = 5
    elif method_name == "bayesian_vi":
        config["vi_samples"] = 20
    return config


def evaluate_method(method, data, benchmark):
    pred = method.predict_with_uncertainty(data.X_test)
    accuracy = np.mean(pred.predictions == data.y_test)
    ece = compute_ece(data.y_test, pred.probabilities)
    bs = brier_score(data.y_test, pred.probabilities)
    y_bayes = benchmark.compute_bayes_optimal(data.X_test)
    bayes_accuracy = np.mean(y_bayes == data.y_test)
    bayes_error = benchmark.get_bayes_error()
    gap = 1 - accuracy - bayes_error
    return {
        "accuracy": accuracy,
        "ece": ece,
        "brier_score": bs,
        "mean_uncertainty": pred.uncertainties.mean(),
        "bayes_accuracy": bayes_accuracy,
        "bayes_gap": gap,
        "predictions": pred,
    }


def cmd_list(args):
    methods = list_methods()
    print("\n" + "=" * 70)
    print("Available UQ Methods")
    print("=" * 70 + "\n")
    for method_name in methods:
        info = method_info(method_name)
        desc = (
            info["docstring"].strip().split("\n")[0]
            if info["docstring"]
            else "No description"
        )
        print(f"  {method_name:20} - {desc}")
    print(f"\n  Total: {len(methods)} methods")
    print("=" * 70 + "\n")


def cmd_train(args):
    print("\n" + "=" * 70)
    print(f"Training: {args.method}")
    print("=" * 70 + "\n")
    (benchmark, data) = create_benchmark(
        train_size=args.train_size, test_size=args.test_size, seed=args.seed
    )
    print(
        f"Train: {len(data.X_train)} | Val: {len(data.X_val)} | Test: {len(data.X_test)}"
    )
    config = get_method_config(args.method, args)
    method = get_method(args.method, config=config)
    metrics = method.fit(data.X_train, data.y_train, data.X_val, data.y_val)
    val_acc = metrics.get("final_val_accuracy", metrics.get("average_val_accuracy", 0))
    train_time = metrics.get("training_time", 0)
    print(f"Completed in {train_time:.1f}s")
    print(f"Validation accuracy: {val_acc:.2f}%")
    results = evaluate_method(method, data, benchmark)
    print("\nResults:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  ECE: {results['ece']:.4f}")
    print(f"  Brier Score: {results['brier_score']:.4f}")
    print(f"  Uncertainty: {results['mean_uncertainty']:.4f}")
    print("\nvs Bayes Optimal:")
    print(f"  Bayes Accuracy: {results['bayes_accuracy']:.4f}")
    print(f"  Performance Gap: {results['bayes_gap']:.4f}")
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        method.save(str(save_path))
        print(f"\nModel saved to: {save_path}")
    print("\n" + "=" * 70 + "\n")


def cmd_compare(args):
    print("\n" + "=" * 70)
    print("Comparing UQ Methods")
    print("=" * 70 + "\n")
    (benchmark, data) = create_benchmark(
        train_size=args.train_size, test_size=args.test_size, seed=args.seed
    )
    bayes_error = benchmark.get_bayes_error()
    print(f"Data generated | Bayes error: {bayes_error:.4f}")

    methods_to_compare = args.methods if args.methods else list_methods()
    results = []
    for i, method_name in enumerate(methods_to_compare, 1):
        print(f"  [{i}/{len(methods_to_compare)}] {method_name}")
        config = get_method_config(method_name, args)
        method = get_method(method_name, config=config)
        metrics = method.fit(data.X_train, data.y_train, data.X_val, data.y_val)
        eval_results = evaluate_method(method, data, benchmark)
        cost = method.get_computational_cost()
        acc = eval_results["accuracy"]
        gap = eval_results["bayes_gap"]
        ece = eval_results["ece"]
        print(f"      Accuracy: {acc:.4f} | Gap: {gap:.4f} | ECE: {ece:.4f}")
        results.append(
            {
                "Method": method_name,
                "Accuracy": eval_results["accuracy"],
                "Bayes Gap": eval_results["bayes_gap"],
                "ECE": eval_results["ece"],
                "Brier": eval_results["brier_score"],
                "Uncertainty": eval_results["mean_uncertainty"],
                "Train Time": metrics.get("training_time", 0),
                "Parameters": cost["total_parameters"],
            }
        )
    print("=" * 70 + "\n")
    df = pd.DataFrame(results)
    print(df.to_string(index=False, float_format="%.4f"))
    print(f"\n{'=' * 70}\n")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian UQ Framework - Train and compare uncertainty quantification methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    subparsers.required = True
    parser_list = subparsers.add_parser("list", help="List available UQ methods")
    parser_list.set_defaults(func=cmd_list)
    parser_train = subparsers.add_parser("train", help="Train a UQ method")
    parser_train.add_argument(
        "method", type=str, help='Method name (see "list" command)'
    )
    parser_train.add_argument(
        "--epochs", type=int, default=100, help="Training epochs (default: 100)"
    )
    parser_train.add_argument(
        "--train-size", type=int, default=500, help="Training set size (default: 500)"
    )
    parser_train.add_argument(
        "--test-size", type=int, default=5000, help="Test set size (default: 5000)"
    )
    parser_train.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser_train.add_argument("--save", type=str, help="Path to save trained model")
    parser_train.set_defaults(func=cmd_train)

    parser_compare = subparsers.add_parser("compare", help="Compare UQ methods")
    parser_compare.add_argument(
        "-m", "--methods", type=str, nargs="+", help="Methods to compare (default: all)"
    )
    parser_compare.add_argument(
        "--epochs", type=int, default=50, help="Training epochs (default: 50)"
    )
    parser_compare.add_argument(
        "--train-size", type=int, default=500, help="Training set size (default: 500)"
    )
    parser_compare.add_argument(
        "--test-size", type=int, default=5000, help="Test set size (default: 5000)"
    )
    parser_compare.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser_compare.add_argument("--output", type=str, help="Path to save results CSV")
    parser_compare.set_defaults(func=cmd_compare)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    args.func(args)


if __name__ == "__main__":
    main()
