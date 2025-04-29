import pandas as pd
import numpy as np
import os
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(ground_truth_path, prediction_path, model_name):
    """
    Calculate metrics comparing ground truth with model predictions
    
    Args:
        ground_truth_path (str): Path to ground truth CSV file
        prediction_path (str): Path to model prediction CSV file
        model_name (str): Name of the model
    
    Returns:
        dict: Dictionary containing all calculated metrics
    """
    # Load data
    gt_df = pd.read_csv(ground_truth_path)
    pred_df = pd.read_csv(prediction_path)
    
    # Merge dataframes on image_name to ensure we're comparing the same images
    merged_df = pd.merge(gt_df, pred_df, on='image_name', suffixes=('_gt', '_pred'))
    
    # Classification metrics for helmet presence
    y_true = merged_df['helmet_presence_gt'].values
    y_pred = merged_df['helmet_presence_pred'].values
    
    # Check if there are both classes in ground truth and prediction
    unique_gt = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    
    # Classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate ROC AUC if possible (need probabilities for proper ROC curve, but do binary classification here)
    roc_auc = roc_auc_score(y_true, y_pred) if (len(unique_gt) > 1 and len(unique_pred) > 1) else None
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Helmet count metrics
    mse = np.mean((merged_df['helmet_count_gt'] - merged_df['helmet_count_pred']) ** 2)
    mae = np.mean(np.abs(merged_df['helmet_count_gt'] - merged_df['helmet_count_pred']))
    rmse = np.sqrt(mse)
    
    # Helmet count accuracy (exact match percentage)
    count_accuracy = np.mean(merged_df['helmet_count_gt'] == merged_df['helmet_count_pred'])
    
    # Create a dataframe to store helmet count differences
    count_diff_df = pd.DataFrame({
        'image_name': merged_df['image_name'],
        'ground_truth': merged_df['helmet_count_gt'],
        'prediction': merged_df['helmet_count_pred'],
        'difference': merged_df['helmet_count_gt'] - merged_df['helmet_count_pred']
    })
    
    # Count positive differences (under predictions) and negative differences (over predictions)
    under_predictions = (count_diff_df['difference'] > 0).sum()
    over_predictions = (count_diff_df['difference'] < 0).sum()
    correct_predictions = (count_diff_df['difference'] == 0).sum()
    
    # Create results dictionary
    results = {
        'model_name': model_name,
        'classification_metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        },
        'count_metrics': {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'count_accuracy': count_accuracy,
            'under_predictions': under_predictions,
            'over_predictions': over_predictions,
            'correct_predictions': correct_predictions,
            'total_images': len(merged_df)
        },
        'raw_data': {
            'merged_df': merged_df,
            'count_diff_df': count_diff_df
        }
    }
    
    return results

def plot_confusion_matrix(cm, model_name, output_dir):
    """
    Plot and save confusion matrix
    
    Args:
        cm: Confusion matrix array
        model_name (str): Name of the model
        output_dir (str): Directory to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Helmet', 'Helmet'],
                yticklabels=['No Helmet', 'Helmet'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()

def plot_count_difference_histogram(count_diff_df, model_name, output_dir):
    """
    Plot and save helmet count difference histogram
    
    Args:
        count_diff_df (DataFrame): DataFrame containing count differences
        model_name (str): Name of the model
        output_dir (str): Directory to save the plot
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(count_diff_df['difference'], kde=True, bins=range(min(count_diff_df['difference'])-1, 
                                                                 max(count_diff_df['difference'])+2))
    plt.title(f'Helmet Count Difference Histogram - {model_name}')
    plt.xlabel('Ground Truth Count - Predicted Count')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add vertical line at x=0 (perfect prediction)
    plt.axvline(x=0, color='red', linestyle='--')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f'{model_name}_count_difference.png'))
    plt.close()

def generate_model_report(results, output_dir):
    """
    Generate a detailed report for a single model
    
    Args:
        results (dict): Dictionary containing all calculated metrics
        output_dir (str): Directory to save the report
    """
    model_name = results['model_name']
    cm = results['classification_metrics']['confusion_matrix']
    count_diff_df = results['raw_data']['count_diff_df']
    
    # Plot and save confusion matrix
    plot_confusion_matrix(cm, model_name, output_dir)
    
    # Plot and save count difference histogram
    plot_count_difference_histogram(count_diff_df, model_name, output_dir)
    
    # Create a markdown report
    report = f"# Model Performance Report: {model_name}\n\n"
    
    # Classification metrics
    report += "## Classification Metrics (Helmet Presence)\n\n"
    report += f"- Accuracy: {results['classification_metrics']['accuracy']:.4f}\n"
    report += f"- Precision: {results['classification_metrics']['precision']:.4f}\n"
    report += f"- Recall: {results['classification_metrics']['recall']:.4f}\n"
    report += f"- F1 Score: {results['classification_metrics']['f1_score']:.4f}\n"
    
    if results['classification_metrics']['roc_auc'] is not None:
        report += f"- ROC AUC: {results['classification_metrics']['roc_auc']:.4f}\n"
    else:
        report += "- ROC AUC: Not available (requires both classes in predictions)\n"
    
    report += "\n### Confusion Matrix\n\n"
    report += "![Confusion Matrix](./"+f"{model_name}_confusion_matrix.png"+")\n\n"
    
    # Count metrics
    report += "## Count Metrics (Number of Helmets)\n\n"
    report += f"- Mean Squared Error (MSE): {results['count_metrics']['mse']:.4f}\n"
    report += f"- Mean Absolute Error (MAE): {results['count_metrics']['mae']:.4f}\n"
    report += f"- Root Mean Squared Error (RMSE): {results['count_metrics']['rmse']:.4f}\n"
    report += f"- Count Accuracy (Exact Match): {results['count_metrics']['count_accuracy']:.4f}\n\n"
    
    report += "### Count Prediction Analysis\n\n"
    report += f"- Total images: {results['count_metrics']['total_images']}\n"
    report += f"- Correct predictions: {results['count_metrics']['correct_predictions']} ({results['count_metrics']['correct_predictions']/results['count_metrics']['total_images']*100:.2f}%)\n"
    report += f"- Under predictions: {results['count_metrics']['under_predictions']} ({results['count_metrics']['under_predictions']/results['count_metrics']['total_images']*100:.2f}%)\n"
    report += f"- Over predictions: {results['count_metrics']['over_predictions']} ({results['count_metrics']['over_predictions']/results['count_metrics']['total_images']*100:.2f}%)\n\n"
    
    report += "### Count Difference Histogram\n\n"
    report += "![Count Difference Histogram](./"+f"{model_name}_count_difference.png"+")\n"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Write the report to a file
    with open(os.path.join(output_dir, f"{model_name}_report.md"), 'w') as f:
        f.write(report)
    
    return report

def compare_models(models_results, output_dir):
    """
    Generate a comparison report for all models
    
    Args:
        models_results (list): List of dictionaries containing results for each model
        output_dir (str): Directory to save the report
    """
    # Create comparison table for classification metrics
    comparison_df = pd.DataFrame([{
        'Model': result['model_name'],
        'Accuracy': result['classification_metrics']['accuracy'],
        'Precision': result['classification_metrics']['precision'],
        'Recall': result['classification_metrics']['recall'],
        'F1 Score': result['classification_metrics']['f1_score'],
        'ROC AUC': result['classification_metrics']['roc_auc'] if result['classification_metrics']['roc_auc'] is not None else np.nan
    } for result in models_results])
    
    # Create comparison table for count metrics
    count_comparison_df = pd.DataFrame([{
        'Model': result['model_name'],
        'MSE': result['count_metrics']['mse'],
        'MAE': result['count_metrics']['mae'],
        'RMSE': result['count_metrics']['rmse'],
        'Count Accuracy': result['count_metrics']['count_accuracy'],
        'Under Predictions %': result['count_metrics']['under_predictions'] / result['count_metrics']['total_images'] * 100,
        'Over Predictions %': result['count_metrics']['over_predictions'] / result['count_metrics']['total_images'] * 100,
        'Correct Predictions %': result['count_metrics']['correct_predictions'] / result['count_metrics']['total_images'] * 100
    } for result in models_results])
    
    # Create comparison report
    report = "# Model Comparison Report\n\n"
    
    # Classification metrics comparison
    report += "## Classification Metrics Comparison\n\n"
    report += comparison_df.to_markdown(index=False) + "\n\n"
    
    # Create a bar chart for classification metrics
    plt.figure(figsize=(12, 6))
    comparison_df_melted = pd.melt(comparison_df, id_vars=['Model'], 
                                   value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                                   var_name='Metric', value_name='Score')
    
    sns.barplot(x='Model', y='Score', hue='Metric', data=comparison_df_melted)
    plt.title('Classification Metrics Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_metrics_comparison.png'))
    plt.close()
    
    report += "![Classification Metrics Comparison](./classification_metrics_comparison.png)\n\n"
    
    # Count metrics comparison
    report += "## Count Metrics Comparison\n\n"
    report += count_comparison_df.to_markdown(index=False) + "\n\n"
    
    # Create a bar chart for MSE, MAE, RMSE
    plt.figure(figsize=(12, 6))
    count_df_melted = pd.melt(count_comparison_df, id_vars=['Model'], 
                              value_vars=['MSE', 'MAE', 'RMSE'],
                              var_name='Metric', value_name='Value')
    
    sns.barplot(x='Model', y='Value', hue='Metric', data=count_df_melted)
    plt.title('Error Metrics Comparison')
    plt.ylabel('Value')
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_metrics_comparison.png'))
    plt.close()
    
    report += "![Error Metrics Comparison](./error_metrics_comparison.png)\n\n"
    
    # Create a bar chart for prediction percentages
    plt.figure(figsize=(12, 6))
    prediction_df_melted = pd.melt(count_comparison_df, id_vars=['Model'], 
                                   value_vars=['Under Predictions %', 'Over Predictions %', 'Correct Predictions %'],
                                   var_name='Type', value_name='Percentage')
    
    sns.barplot(x='Model', y='Percentage', hue='Type', data=prediction_df_melted)
    plt.title('Prediction Type Distribution')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 100)
    plt.legend(title='Prediction Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_distribution_comparison.png'))
    plt.close()
    
    report += "![Prediction Distribution Comparison](./prediction_distribution_comparison.png)\n\n"
    
    # Create a summary ranking table
    # For each metric, rank the models (1 is best)
    # For accuracy, precision, recall, f1, roc_auc, count_accuracy, correct_predictions: higher is better
    # For MSE, MAE, RMSE: lower is better
    
    # Rank for classification metrics (higher is better)
    ranking_df = pd.DataFrame()
    ranking_df['Model'] = comparison_df['Model']
    
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']:
        if metric in comparison_df.columns:
            ranking_df[f'{metric} Rank'] = comparison_df[metric].rank(ascending=False)
    
    # Rank for count metrics
    for metric in ['MSE', 'MAE', 'RMSE']:
        ranking_df[f'{metric} Rank'] = count_comparison_df[metric].rank(ascending=True)
    
    ranking_df['Count Accuracy Rank'] = count_comparison_df['Count Accuracy'].rank(ascending=False)
    ranking_df['Correct Predictions % Rank'] = count_comparison_df['Correct Predictions %'].rank(ascending=False)
    
    # Calculate average rank
    rank_columns = [col for col in ranking_df.columns if 'Rank' in col]
    ranking_df['Average Rank'] = ranking_df[rank_columns].mean(axis=1)
    
    # Sort by average rank
    ranking_df = ranking_df.sort_values('Average Rank')
    
    report += "## Model Rankings (1 = Best)\n\n"
    report += ranking_df.to_markdown(index=False) + "\n\n"
    
    # Add conclusion
    best_model = ranking_df.iloc[0]['Model']
    report += "## Conclusion\n\n"
    report += f"Based on the average ranking across all metrics, **{best_model}** performs the best overall.\n\n"
    
    # Write the report to a file
    with open(os.path.join(output_dir, "models_comparison_report.md"), 'w') as f:
        f.write(report)
    
    return report

def main():
    parser = argparse.ArgumentParser(description='Evaluate helmet detection models')
    parser.add_argument('--ground_truth', type=str, required=True, help='Path to ground truth CSV file')
    parser.add_argument('--predictions', type=str, nargs='+', required=True, help='Paths to prediction CSV files')
    parser.add_argument('--model_names', type=str, nargs='+', required=True, help='Names of the models')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Check if number of model names matches number of prediction files
    if len(args.predictions) != len(args.model_names):
        raise ValueError("Number of prediction files must match number of model names")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each model
    all_results = []
    for pred_file, model_name in zip(args.predictions, args.model_names):
        print(f"Evaluating model: {model_name}")
        results = calculate_metrics(args.ground_truth, pred_file, model_name)
        all_results.append(results)
        
        # Generate individual model report
        generate_model_report(results, args.output_dir)
    
    # Generate comparison report
    compare_models(all_results, args.output_dir)
    
    print(f"Evaluation completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()