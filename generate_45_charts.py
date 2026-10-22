import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import itertools

# Create a folder specifically to hold the 45 charts
output_dir = 'analytical_charts'
os.makedirs(output_dir, exist_ok=True)

print("Generating simulated dataset for 100 enterprise prompts...")
np.random.seed(42)
domains = ['Enterprise Networking', 'Cybersecurity', 'Cloud Infrastructure', 'Telecom Telemetry', 'General Reasoning']
complexities = ['Low', 'Medium', 'High']

data = {
    'prompt_id': range(1, 101),
    'domain': np.random.choice(domains, 100),
    'complexity': np.random.choice(complexities, 100, p=[0.2, 0.5, 0.3]),
    'model_a_score': np.random.randint(2, 6, 100),
    'model_b_score': np.random.randint(1, 6, 100),
    'factual_accuracy_a': np.random.uniform(3.0, 5.0, 100),
    'factual_accuracy_b': np.random.uniform(1.0, 5.0, 100),
    'config_correctness_a': np.random.uniform(2.0, 5.0, 100),
    'config_correctness_b': np.random.uniform(1.0, 4.5, 100),
    'hallucination_a': np.random.choice([0, 1], 100, p=[0.85, 0.15]), # 0=False, 1=True
    'hallucination_b': np.random.choice([0, 1], 100, p=[0.60, 0.40]),
    'evaluator_confidence': np.random.uniform(0.75, 1.0, 100)
}
df = pd.DataFrame(data)
df['win_margin'] = df['model_a_score'] - df['model_b_score']

print("Initializing 45 unique charts generation pipeline. LET'S GO!")
chart_counter = 1

def save_chart(name):
    global chart_counter
    filename = f"{output_dir}/chart_{str(chart_counter).zfill(2)}_{name}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    chart_counter += 1

sns.set_theme(style="darkgrid", palette="muted")

# --- BATCH 1: Distributions & Histograms (Charts 1-5) ---
sns.histplot(df['model_a_score'], bins=5, kde=True, color='blue'); plt.title('Model A Score Distribution'); save_chart('hist_model_a')
sns.histplot(df['model_b_score'], bins=5, kde=True, color='red'); plt.title('Model B Score Distribution'); save_chart('hist_model_b')
sns.kdeplot(data=df, x='factual_accuracy_a', hue='domain', fill=True); plt.title('Factual Accuracy A by Domain'); save_chart('kde_factual_a')
sns.kdeplot(data=df, x='factual_accuracy_b', hue='domain', fill=True); plt.title('Factual Accuracy B by Domain'); save_chart('kde_factual_b')
sns.histplot(df['evaluator_confidence'], bins=10, color='purple'); plt.title('Evaluator Confidence Histogram'); save_chart('hist_confidence')

# --- BATCH 2: Categorical Comparisons (Charts 6-15) ---
for metric in ['model_a_score', 'model_b_score', 'factual_accuracy_a', 'factual_accuracy_b', 'config_correctness_a']:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='domain', y=metric, data=df); plt.title(f'Boxplot: {metric} by Domain'); save_chart(f'box_{metric}')
    plt.figure(figsize=(10, 5))
    sns.violinplot(x='complexity', y=metric, data=df, split=True); plt.title(f'Violin: {metric} by Complexity'); save_chart(f'violin_{metric}')

# --- BATCH 3: Scatter & Relationships (Charts 16-25) ---
metrics_pairs = list(itertools.combinations(['factual_accuracy_a', 'config_correctness_a', 'evaluator_confidence', 'win_margin', 'factual_accuracy_b'], 2))
for x_met, y_met in metrics_pairs[:10]:
    sns.scatterplot(x=x_met, y=y_met, hue='domain', size='complexity', data=df)
    plt.title(f'Scatter: {x_met} vs {y_met}'); save_chart(f'scatter_{x_met}_vs_{y_met}')

# --- BATCH 4: Advanced Statistical Plots (Charts 26-35) ---
for col in ['model_a_score', 'model_b_score', 'factual_accuracy_a', 'config_correctness_a', 'win_margin']:
    plt.figure(figsize=(10, 5))
    sns.swarmplot(x='domain', y=col, data=df, size=4); plt.title(f'Swarmplot: {col} by Domain'); save_chart(f'swarm_{col}')
    sns.ecdfplot(data=df, x=col, hue='complexity'); plt.title(f'ECDF: {col} by Complexity'); save_chart(f'ecdf_{col}')

# --- BATCH 5: Heatmaps, Pies, and Density (Charts 36-45) ---
# 36. Correlation Heatmap
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f"); plt.title('Numerical Correlation Heatmap'); save_chart('heatmap_correlation')

# 37-38. Hexbin Density
plt.hexbin(df['factual_accuracy_a'], df['config_correctness_a'], gridsize=15, cmap='Blues'); plt.title('Hexbin: Fact vs Config (A)'); save_chart('hexbin_A')
plt.hexbin(df['factual_accuracy_b'], df['config_correctness_b'], gridsize=15, cmap='Reds'); plt.title('Hexbin: Fact vs Config (B)'); save_chart('hexbin_B')

# 39-40. Pie Charts for Hallucinations
halluc_a_counts = df['hallucination_a'].value_counts()
plt.pie(halluc_a_counts, labels=['Clean', 'Hallucinated'], autopct='%1.1f%%', colors=['#4CAF50', '#F44336']); plt.title('Model A Hallucination Rate'); save_chart('pie_halluc_A')
halluc_b_counts = df['hallucination_b'].value_counts()
plt.pie(halluc_b_counts, labels=['Clean', 'Hallucinated'], autopct='%1.1f%%', colors=['#4CAF50', '#F44336']); plt.title('Model B Hallucination Rate'); save_chart('pie_halluc_B')

# 41-43. Barplots with Error Bars
sns.barplot(x='domain', y='win_margin', data=df, errorbar='sd'); plt.title('Win Margin by Domain (w/ StdDev)'); save_chart('bar_win_margin')
sns.barplot(x='complexity', y='evaluator_confidence', data=df, errorbar='ci'); plt.title('Confidence by Complexity'); save_chart('bar_confidence')
sns.pointplot(x='domain', y='factual_accuracy_a', hue='complexity', data=df); plt.title('Pointplot: Accuracy trends'); save_chart('point_accuracy')

# 44-45. Joint Plots
sns.jointplot(x='factual_accuracy_a', y='model_a_score', data=df, kind="reg"); plt.suptitle('Jointplot: Accuracy vs Score (A)', y=1.02); save_chart('joint_A')
sns.jointplot(x='factual_accuracy_b', y='model_b_score', data=df, kind="hex"); plt.suptitle('Jointplot: Accuracy vs Score (B)', y=1.02); save_chart('joint_B')

print(f"COMPLETE! {chart_counter - 1} unique, highly technical charts have been generated in the 'analytical_charts' folder.")
