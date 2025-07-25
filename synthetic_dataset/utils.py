import json
import matplotlib.pyplot as plt
import pandas as pd
from deepeval.evaluate.types import EvaluationResult


def print_eval_results(res):
    print(f"Evaluation results (in total: {len(res.test_results)})")
    for test_res in res.test_results:
        print(f"    TestCaseName: {test_res.name}")
        print(f"    Success: {test_res.success}")
        print(f"    Input: {test_res.input}")
        print(f"    Actual Output: {test_res.actual_output}")
        print(f"    Metrics data of {len(test_res.metrics_data)} metrics:")
        for mdata in test_res.metrics_data:
            print(f"        Metric Name: {mdata.name}, Success: {mdata.success}, Score: {mdata.score}, "
                f"Cost: {mdata.evaluation_cost}, Reason: {mdata.reason}")
        print("********************************************\n")

def results_to_df(res: EvaluationResult, json_file_path: str) -> pd.DataFrame:
    """Convert evaluation results to a pandas DataFrame."""
    rows = []
    for test_res in res.test_results:
        if test_res.metrics_data:  # Check if metrics_data is not None
            for mdata in test_res.metrics_data:
                row = {
                    'test_case_name': test_res.name,
                    'test_success': test_res.success,
                    'input': test_res.input,
                    'actual_output': test_res.actual_output,
                    'metric_name': mdata.name,
                    'metric_success': mdata.success,
                    'metric_score': mdata.score,
                    'metric_threshold': mdata.threshold,
                    'evaluation_cost': mdata.evaluation_cost,
                    'evaluation_model': mdata.evaluation_model,
                    'reason': mdata.reason
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)

    metric_summary = df.groupby('metric_name').agg({
        'metric_score': ['mean', 'std', 'min', 'max'],
        'metric_success': 'mean',
        'evaluation_cost': 'sum'
    }).round(3)
    results_summary = {
        'total_test_cases': int(df['test_case_name'].nunique()),
        'total_metrics_evaluated': len(df),
        'average_score_all_metrics': round(df['metric_score'].mean(), 3),
        'overall_success_rate': round(df['metric_success'].mean(), 3),
        'metric_summary': {
            metric_name: {
                'metric_score_mean': stats[('metric_score', 'mean')],
                'metric_score_std': stats[('metric_score', 'std')],
                'metric_score_min': stats[('metric_score', 'min')],
                'metric_score_max': stats[('metric_score', 'max')],
                'metric_success_rate': stats[('metric_success', 'mean')],
                'evaluation_cost_total': stats[('evaluation_cost', 'sum')]
            }
            for metric_name, stats in metric_summary.iterrows()
        }
    }

    if json_file_path:
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(rows, f, indent=4, ensure_ascii=False)
        
        json_output_path = json_file_path.replace('.json', '_summary.json')
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=4, ensure_ascii=False)
    return df

def visualize_eval_results(df: pd.DataFrame, eval_results_figure_path: str):
    # Set up the plotting style
    plt.style.use('default')
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('DeepEval Metrics Analysis', fontsize=16, fontweight='bold')
    
    # 1. Metric Scores by Test Case
    df.pivot_table(index='test_case_name', columns='metric_name', 
                   values='metric_score').plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Metric Scores by Test Case')
    axes[0,0].set_ylabel('Score')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Success Rate by Metric
    success_rate = df.groupby('metric_name')['metric_success'].mean()
    success_rate.plot(kind='bar', ax=axes[0,1], color='lightgreen')
    axes[0,1].set_title('Success Rate by Metric')
    axes[0,1].set_ylabel('Success Rate')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Distribution of Metric Scores
    for metric in df['metric_name'].unique():
        metric_scores = df[df['metric_name'] == metric]['metric_score']
        axes[1,0].hist(metric_scores, alpha=0.7, label=metric, bins=10)
    axes[1,0].set_title('Distribution of Metric Scores')
    axes[1,0].set_xlabel('Score')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].legend()
    
    # 4. Evaluation Cost by Metric
    if df['evaluation_cost'].notna().any():
        cost_by_metric = df.groupby('metric_name')['evaluation_cost'].sum()
        cost_by_metric.plot(kind='pie', ax=axes[1,1], autopct='%1.1f%%')
        axes[1,1].set_title('Evaluation Cost Distribution')
    else:
        axes[1,1].text(0.5, 0.5, 'No cost data available', 
                       ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Evaluation Cost Distribution')
    
    plt.tight_layout()
    plt.savefig(eval_results_figure_path, dpi=300, bbox_inches='tight')
    # plt.show()  # uncomment to display the plots interactively
