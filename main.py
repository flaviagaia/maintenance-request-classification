from src.pipeline import run_pipeline


if __name__ == "__main__":
    summary = run_pipeline()
    print("Maintenance Request Classification")
    print("-" * 48)
    print(f"Rows: {summary['rows']}")
    print(f"Maintenance groups: {summary['maintenance_groups']}")
    print(f"Best model: {summary['best_model']}")
    print(f"Validation macro F1: {summary['best_macro_f1']:.4f}")
    print(f"Validation accuracy: {summary['best_accuracy']:.4f}")
