import matplotlib.pyplot as plt
import pandas as pd


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

def results_to_df(res: dict) -> pd.DataFrame:
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
                    'evaluation_cost': mdata.evaluation_cost,
                    'reason': mdata.reason
                }
                rows.append(row)
    return pd.DataFrame(rows)


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
    
    # Additional detailed analysis
    print("\nDetailed Analysis:")
    print("=" * 50)
    print(f"Total test cases: {df['test_case_name'].nunique()}")
    print(f"Total metrics evaluated: {len(df)}")
    print(f"Average score across all metrics: {df['metric_score'].mean():.3f}")
    print(f"Overall success rate: {df['metric_success'].mean():.3f}")
    
    print("\nMetric Summary:")
    metric_summary = df.groupby('metric_name').agg({
        'metric_score': ['mean', 'std', 'min', 'max'],
        'metric_success': 'mean',
        'evaluation_cost': 'sum'
    }).round(3)
    print(metric_summary)
