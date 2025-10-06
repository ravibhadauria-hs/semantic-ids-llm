#!/usr/bin/env python3
"""
Semantic ID Evaluation Script
Samples jobs that match at the given semantic level and generates clickable Handshake URLs.
"""

import polars as pl
import numpy as np
import random
from collections import Counter
from pathlib import Path

def load_semantic_ids_data(file_path: str) -> pl.DataFrame:
    """Load semantic IDs data from parquet file."""
    return pl.read_parquet(file_path)

def find_common_combinations_by_level(df: pl.DataFrame, level: int, min_count: int = 10) -> pl.DataFrame:
    """Find semantic ID combinations up to the specified level that have at least min_count jobs.
    
    Args:
        df: DataFrame with semantic IDs
        level: Level to group by (0=first level only, 1=first two levels, etc.)
        min_count: Minimum number of jobs required for a combination to be considered
    """
    # Dynamically determine available levels from column names
    available_levels = [col for col in df.columns if col.startswith("semantic_id_level_")]
    max_level = len(available_levels) - 1  # 0-indexed
    
    if level < 0 or level > max_level:
        raise ValueError(f"Level must be between 0 and {max_level} (inclusive). Available levels: {list(range(max_level + 1))}")
    
    # Build group columns up to the specified level
    group_cols = [f"semantic_id_level_{i}" for i in range(level + 1)]
    
    level_counts = df.group_by(group_cols).agg(pl.len().alias("count")).sort("count", descending=True)
    common_combinations = level_counts.filter(pl.col("count") >= min_count)
    return common_combinations

def sample_jobs_by_level(df: pl.DataFrame, level: int, combination_values: list, sample_size: int = 10) -> pl.DataFrame:
    """Sample jobs that have the specified semantic ID combination up to the given level.
    
    Args:
        df: DataFrame with semantic IDs
        level: Level to match (0=first level only, 1=first two levels, etc.)
        combination_values: List of values to match [level_0, level_1, ...] (unused levels ignored)
        sample_size: Number of jobs to sample
    """
    # Dynamically determine available levels from column names
    available_levels = [col for col in df.columns if col.startswith("semantic_id_level_")]
    max_level = len(available_levels) - 1  # 0-indexed
    
    if level < 0 or level > max_level:
        raise ValueError(f"Level must be between 0 and {max_level} (inclusive). Available levels: {list(range(max_level + 1))}")
    
    # Build filter conditions for all levels up to the specified level
    conditions = []
    for i in range(level + 1):
        if i < len(combination_values):
            conditions.append(pl.col(f"semantic_id_level_{i}") == combination_values[i])
    
    # Apply all conditions
    if len(conditions) == 1:
        matching_jobs = df.filter(conditions[0])
    else:
        combined_condition = conditions[0]
        for condition in conditions[1:]:
            combined_condition = combined_condition & condition
        matching_jobs = df.filter(combined_condition)
    
    if len(matching_jobs) < sample_size:
        sample_size = len(matching_jobs)
    
    # Random sampling
    sampled_indices = random.sample(range(len(matching_jobs)), sample_size)
    return matching_jobs[sampled_indices]

def generate_handshake_urls(job_ids: list) -> list:
    """Generate clickable Handshake URLs for job IDs."""
    base_url = "https://app.joinhandshake.com/jobs/"
    return [f"{base_url}{job_id}" for job_id in job_ids]

def create_evaluation_report(df: pl.DataFrame, level: int = 2, num_semantic_ids: int = 5, sample_size: int = 10) -> str:
    """Create a comprehensive evaluation report for the specified hierarchical level.
    
    Args:
        df: DataFrame with semantic IDs
        level: Level to evaluate (0=first level only, 1=first two levels, etc.)
        num_semantic_ids: Number of semantic ID groups to evaluate
        sample_size: Number of jobs per group
    """
    
    # Dynamically determine available levels
    available_levels = [col for col in df.columns if col.startswith("semantic_id_level_")]
    max_level = len(available_levels) - 1  # 0-indexed
    
    if level < 0 or level > max_level:
        raise ValueError(f"Level must be between 0 and {max_level} (inclusive). Available levels: {list(range(max_level + 1))}")
    
    # Find common combinations at the specified level
    common_combinations = find_common_combinations_by_level(df, level, min_count=sample_size)
    
    if len(common_combinations) < num_semantic_ids:
        num_semantic_ids = len(common_combinations)
    
    # Select combinations from common ones
    selected_combinations = common_combinations.head(num_semantic_ids)
    
    # Determine level description dynamically
    if level == 0:
        level_description = "first level only (Level 0)"
    elif level == max_level:
        level_description = f"all {max_level + 1} levels (Levels 0-{max_level})"
    else:
        level_description = f"first {level + 1} levels (Levels 0-{level})"
    
    report = []
    report.append("# Semantic ID Evaluation Report")
    report.append("")
    report.append(f"This report evaluates the semantic clustering quality by examining jobs that share the same semantic ID at the {level_description}.")
    report.append("")
    report.append(f"**Dataset**: {len(df):,} total jobs")
    report.append(f"**Evaluation Level**: {level_description}")
    report.append(f"**Evaluation**: {num_semantic_ids} semantic ID groups")
    report.append(f"**Sample Size**: {sample_size} jobs per group")
    report.append("")
    
    for i, row in enumerate(selected_combinations.iter_rows(), 1):
        # Dynamically extract values based on the level
        combination_values = [row[j] for j in range(level + 1)]
        total_jobs = row[level + 1]  # The count is always the last column
        
        # Create combination string
        combination_str = "-".join(map(str, combination_values))
        
        report.append(f"## Semantic ID Group {i}")
        report.append(f"**Semantic ID Code**: {combination_str}")
        report.append(f"**Total Jobs**: {total_jobs:,}")
        
        # Sample jobs
        sampled_jobs = sample_jobs_by_level(df, level, combination_values, sample_size)
        job_ids = sampled_jobs["parent_asin"].to_list()
        handshake_urls = generate_handshake_urls(job_ids)
        
        report.append(f"**Sampled Jobs** ({len(job_ids)} jobs):")
        report.append("")
        
        for j, url in enumerate(handshake_urls, 1):
            report.append(f"{j}. {url}")
        
        # Show semantic ID breakdown for sampled jobs
        report.append("")
        report.append("**Semantic ID Breakdown for Sampled Jobs**:")
        report.append("")
        
        # Build table header dynamically
        header_cols = ["Job ID"] + [f"Level {i}" for i in range(max_level + 1)] + ["Full Semantic ID"]
        header_row = "| " + " | ".join(header_cols) + " |"
        separator_row = "|" + "|".join(["--------" for _ in header_cols]) + "|"
        
        report.append(header_row)
        report.append(separator_row)
        
        for job in sampled_jobs.iter_rows():
            job_id = job[0]
            level_values = [job[i + 1] for i in range(max_level + 1)]  # Skip job_id, get all level values
            full_id = job[-1]  # Full semantic ID is always the last column
            
            # Build table row
            row_values = [str(job_id)] + [str(v) for v in level_values] + [str(full_id)]
            row = "| " + " | ".join(row_values) + " |"
            report.append(row)
        
        report.append("")
        report.append("---")
        report.append("")
    
    # Add summary statistics
    report.append("## Summary Statistics")
    report.append("")
    
    # Get combinations at the specified level
    group_cols = [f"semantic_id_level_{i}" for i in range(level + 1)]
    
    level_combinations = df.group_by(group_cols).agg(pl.len().alias("count"))
    total_unique_combinations = len(level_combinations)
    report.append(f"- **Total unique {level_description} combinations**: {total_unique_combinations:,}")
    
    # Distribution of combinations
    level_dist = level_combinations.sort("count", descending=True)
    
    # Build most/least common combination strings dynamically
    most_common_values = [level_dist[0, i] for i in range(level + 1)]
    least_common_values = [level_dist[-1, i] for i in range(level + 1)]
    most_common_str = "-".join(map(str, most_common_values))
    least_common_str = "-".join(map(str, least_common_values))
    
    most_common_count = level_dist[0, level + 1]  # Count is always at position level + 1
    least_common_count = level_dist[-1, level + 1]
    
    report.append(f"- **Most common combination**: {most_common_str} ({most_common_count:,} jobs)")
    report.append(f"- **Least common combination**: {least_common_str} ({least_common_count:,} jobs)")
    
    # Average jobs per combination
    avg_jobs_per_combination = len(df) / total_unique_combinations
    report.append(f"- **Average jobs per {level_description} combination**: {avg_jobs_per_combination:.1f}")
    
    # Show distribution of combination sizes
    combination_size_dist = level_combinations.group_by("count").agg(pl.len().alias("n_combinations")).sort("count")
    report.append("")
    report.append("**Distribution of combination sizes**:")
    report.append("| Jobs per Combination | Number of Combinations |")
    report.append("|---------------------|----------------------|")
    for row in combination_size_dist.iter_rows():
        jobs_per_combo, n_combos = row[0], row[1]
        report.append(f"| {jobs_per_combo} | {n_combos:,} |")
    
    report.append("")
    report.append("## Evaluation Instructions")
    report.append("")
    report.append("1. **Click on the Handshake URLs** above to view the actual job postings")
    report.append("2. **Evaluate semantic similarity** - Do the jobs in each group seem semantically related?")
    report.append("3. **Check for consistency** - Are the job types, industries, or requirements similar?")
    report.append("4. **Note any outliers** - Are there jobs that don't seem to belong in their group?")
    report.append("5. **Assess clustering quality** - How well does the complete three-level semantic ID capture job similarity?")
    report.append("")
    report.append("## Expected Results")
    report.append("")
    report.append("- **Good clustering**: Jobs with identical three-level semantic IDs should be highly semantically similar")
    report.append("- **Consistent patterns**: Jobs should share common themes like job function, seniority level, or industry")
    report.append("- **Minimal outliers**: Most jobs in a group should be clearly related")
    report.append("- **Hierarchical structure**: The three levels should represent increasingly specific job characteristics")
    report.append("")
    
    return "\n".join(report)

def modify_evaluation_output_path(data_file: str, level: int, output: str) -> str:
    """Modify evaluation output path to include config information from data file.
    
    Args:
        data_file: Path to semantic IDs data file
        level: Evaluation level
        output: Original output path
        
    Returns:
        Modified output path with config info
    """
    from pathlib import Path
    
    # Try to extract config info from data file
    try:
        df = pl.read_parquet(data_file)
        if "codebook_levels" in df.columns and "codebook_size" in df.columns:
            levels = df["codebook_levels"][0]
            codebook_size = df["codebook_size"][0]
            collision_resolved = df["collision_resolved"][0] if "collision_resolved" in df.columns else False
            
            # Parse the output path
            path = Path(output)
            base_name = path.stem
            extension = path.suffix
            
            # Add config information to filename
            collision_suffix = "_collision_resolved" if collision_resolved else ""
            new_filename = f"{base_name}_L{levels}_C{codebook_size}_level{level}{collision_suffix}{extension}"
            
            # Create new path and ensure directory exists
            new_path = path.parent / new_filename
            new_path.parent.mkdir(parents=True, exist_ok=True)
            return str(new_path)
    except Exception as e:
        print(f"Warning: Could not extract config from data file: {e}")
    
    # Fallback to original output path
    return output


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate semantic ID clustering quality at different hierarchical levels")
    parser.add_argument("--level", type=int, default=2, 
                       help="Hierarchical level to evaluate (0=first level only, 1=first two levels, etc.)")
    parser.add_argument("--num-groups", type=int, default=5, 
                       help="Number of semantic ID groups to evaluate")
    parser.add_argument("--sample-size", type=int, default=10, 
                       help="Number of jobs to sample per group")
    parser.add_argument("--data-file", type=str, default="data/output/jobs_semantic_ids.parquet",
                       help="Path to semantic IDs data file")
    parser.add_argument("--output", type=str, default="evaluation_reports/SEMANTIC_ID_EVALUATION.md",
                       help="Output file for evaluation report")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Load data
    if not Path(args.data_file).exists():
        print(f"Error: Data file {args.data_file} not found")
        return
    
    print(f"Loading semantic IDs data from {args.data_file}...")
    df = load_semantic_ids_data(args.data_file)
    print(f"Loaded {len(df):,} jobs")
    
    # Modify output path to include config information
    modified_output = modify_evaluation_output_path(args.data_file, args.level, args.output)
    print(f"Output will be saved to: {modified_output}")
    
    # Create evaluation report
    # Determine level description dynamically
    available_levels = [col for col in df.columns if col.startswith("semantic_id_level_")]
    max_level = len(available_levels) - 1
    
    if args.level == 0:
        level_desc = "first level only"
    elif args.level == max_level:
        level_desc = f"all {max_level + 1} levels"
    else:
        level_desc = f"first {args.level + 1} levels"
    
    print(f"Generating evaluation report for {level_desc}...")
    report = create_evaluation_report(df, level=args.level, num_semantic_ids=args.num_groups, sample_size=args.sample_size)
    
    # Save report
    with open(modified_output, "w") as f:
        f.write(report)
    
    print(f"Evaluation report saved to {modified_output}")
    print(f"Report contains {args.num_groups} semantic ID groups with clickable Handshake URLs")
    print(f"Evaluating {level_desc} semantic ID matches")

if __name__ == "__main__":
    main()
