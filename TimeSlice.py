"""
Time-Slice Validation Method

This alternative validation approach divides each day into 5-minute slices and compares
ground truth and predictions for each slice, providing a more continuous evaluation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_time_slices(ground_truth_df, predictions_df, interval_minutes=5):
    """
    Create time slices for each user and day

    Parameters:
    ground_truth_df (pandas.DataFrame): Ground truth DataFrame
    predictions_df (pandas.DataFrame): Predictions DataFrame
    interval_minutes (int): Size of time intervals in minutes

    Returns:
    dict: Dictionary of time slices by user and day
    """
    # Ensure datetime columns
    date_cols = ['startTime', 'endTime']
    for col in date_cols:
        if col in ground_truth_df.columns and not pd.api.types.is_datetime64_dtype(ground_truth_df[col]):
            ground_truth_df[col] = pd.to_datetime(ground_truth_df[col], errors='coerce')
        if col in predictions_df.columns and not pd.api.types.is_datetime64_dtype(predictions_df[col]):
            predictions_df[col] = pd.to_datetime(predictions_df[col], errors='coerce')

    # Filter out rows with invalid dates
    ground_truth_df = ground_truth_df.dropna(subset=['startTime'])
    predictions_df = predictions_df.dropna(subset=['startTime'])

    # Create user-day dictionary
    user_days = {}

    # Process ground truth data
    for email, user_gt in ground_truth_df.groupby('email'):
        if email not in user_days:
            user_days[email] = {}

        # Extract unique days
        user_gt['date'] = user_gt['startTime'].dt.date
        for date, day_gt in user_gt.groupby('date'):
            # Skip days without any data
            if day_gt.empty:
                continue

            # Create day key
            day_key = date.strftime('%Y-%m-%d')

            # Initialize day if not exists
            if day_key not in user_days[email]:
                user_days[email][day_key] = {
                    'ground_truth': [],
                    'predictions': []
                }

            # Add ground truth entries
            user_days[email][day_key]['ground_truth'].extend(day_gt.to_dict('records'))

    # Process predictions data
    for email, user_pred in predictions_df.groupby('email'):
        if email not in user_days:
            continue  # Skip users not in ground truth

        # Extract unique days
        user_pred['date'] = user_pred['startTime'].dt.date
        for date, day_pred in user_pred.groupby('date'):
            # Skip days without any data
            if day_pred.empty:
                continue

            # Create day key
            day_key = date.strftime('%Y-%m-%d')

            # Skip days not in ground truth
            if day_key not in user_days[email]:
                continue

            # Add prediction entries
            user_days[email][day_key]['predictions'].extend(day_pred.to_dict('records'))

    # Create time slices for each day
    time_slices = {}

    for email, days in user_days.items():
        time_slices[email] = {}

        for day_key, day_data in days.items():
            # Skip days with no ground truth or predictions
            if not day_data['ground_truth'] or not day_data['predictions']:
                continue

            # Convert day_key to datetime
            try:
                day_date = datetime.strptime(day_key, '%Y-%m-%d').date()
            except ValueError:
                continue

            # Find earliest start and latest end for the day
            gt_starts = [entry['startTime'] for entry in day_data['ground_truth'] if
                         isinstance(entry['startTime'], pd.Timestamp)]
            gt_ends = [entry['endTime'] for entry in day_data['ground_truth'] if
                       isinstance(entry.get('endTime'), pd.Timestamp)]

            pred_starts = [entry['startTime'] for entry in day_data['predictions'] if
                           isinstance(entry['startTime'], pd.Timestamp)]
            pred_ends = [entry['endTime'] for entry in day_data['predictions'] if
                         isinstance(entry.get('endTime'), pd.Timestamp)]

            all_times = gt_starts + gt_ends + pred_starts + pred_ends
            all_times = [t for t in all_times if t is not None and not pd.isna(t)]

            if not all_times:
                continue

            day_start = min(all_times).replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = min(all_times).replace(hour=23, minute=59, second=59, microsecond=999999)

            # Create time slices
            time_slices[email][day_key] = []
            current_time = day_start

            while current_time < day_end:
                slice_end = current_time + timedelta(minutes=interval_minutes)

                # Find ground truth for this slice
                gt_segment = None
                gt_mode = None
                gt_purpose = None

                for entry in day_data['ground_truth']:
                    entry_start = entry['startTime']
                    entry_end = entry.get('endTime')

                    # Skip entries with invalid dates
                    if not isinstance(entry_start, pd.Timestamp):
                        continue

                    # Handle missing end time
                    if entry_end is None or pd.isna(entry_end):
                        # Use next entry's start time or day end
                        later_entries = [e for e in day_data['ground_truth']
                                         if isinstance(e['startTime'], pd.Timestamp) and e['startTime'] > entry_start]
                        if later_entries:
                            entry_end = min(e['startTime'] for e in later_entries)
                        else:
                            entry_end = day_end

                    # Check if slice overlaps with entry
                    if entry_start <= slice_end and current_time <= entry_end:
                        gt_segment = entry.get('segment')
                        gt_mode = entry.get('mode') if gt_segment == 'moving' else None
                        gt_purpose = entry.get('purpose') if gt_segment == 'stationary' else None

                        # Use normalized values if available
                        if 'normalized_mode' in entry:
                            gt_mode = entry['normalized_mode']
                        if 'normalized_purpose' in entry:
                            gt_purpose = entry['normalized_purpose']

                        break

                # Find prediction for this slice
                pred_segment = None
                pred_mode = None
                pred_purpose = None

                for entry in day_data['predictions']:
                    entry_start = entry['startTime']
                    entry_end = entry.get('endTime')

                    # Skip entries with invalid dates
                    if not isinstance(entry_start, pd.Timestamp):
                        continue

                    # Handle missing end time
                    if entry_end is None or pd.isna(entry_end):
                        # Use next entry's start time or day end
                        later_entries = [e for e in day_data['predictions']
                                         if isinstance(e['startTime'], pd.Timestamp) and e['startTime'] > entry_start]
                        if later_entries:
                            entry_end = min(e['startTime'] for e in later_entries)
                        else:
                            entry_end = day_end

                    # Check if slice overlaps with entry
                    if entry_start <= slice_end and current_time <= entry_end:
                        pred_segment = entry.get('segment')
                        pred_mode = entry.get('mode') if pred_segment == 'moving' else None
                        pred_purpose = entry.get('purpose') if pred_segment == 'stationary' else None

                        # Use normalized values if available
                        if 'normalized_mode' in entry:
                            pred_mode = entry['normalized_mode']
                        if 'normalized_purpose' in entry:
                            pred_purpose = entry['normalized_purpose']

                        # Extract mode from JSON if needed
                        if pred_mode and isinstance(pred_mode, str) and (
                                pred_mode.startswith('{') or pred_mode.startswith('[')):
                            try:
                                import json
                                mode_dict = json.loads(pred_mode.replace("'", '"'))
                                if isinstance(mode_dict, dict):
                                    pred_mode = max(mode_dict.items(), key=lambda x: float(x[1]))[0]
                            except:
                                pass

                        break

                # Create time slice
                time_slice = {
                    'email': email,
                    'day': day_key,
                    'slice_start': current_time,
                    'slice_end': slice_end,
                    'gt_segment': gt_segment,
                    'gt_mode': gt_mode,
                    'gt_purpose': gt_purpose,
                    'pred_segment': pred_segment,
                    'pred_mode': pred_mode,
                    'pred_purpose': pred_purpose,
                    'segment_match': gt_segment == pred_segment if gt_segment and pred_segment else None,
                    'mode_match': gt_mode == pred_mode if gt_mode and pred_mode else None,
                    'purpose_match': gt_purpose == pred_purpose if gt_purpose and pred_purpose else None
                }

                time_slices[email][day_key].append(time_slice)

                # Move to next slice
                current_time = slice_end

    return time_slices


def calculate_timeslice_metrics(time_slices):
    """
    Calculate validation metrics based on time slices

    Parameters:
    time_slices (dict): Dictionary of time slices by user and day

    Returns:
    dict: Dictionary with calculated metrics
    """
    # Flatten time slices
    all_slices = []
    for email in time_slices:
        for day in time_slices[email]:
            all_slices.extend(time_slices[email][day])

    # Convert to DataFrame
    slices_df = pd.DataFrame(all_slices)

    # Skip if no data
    if slices_df.empty:
        return {
            'segment_accuracy': 0,
            'mode_accuracy': 0,
            'purpose_accuracy': 0,
            'home_work_purpose_accuracy': 0,
            'time_coverage': 0,
            'per_user': {}
        }

    # Calculate metrics
    metrics = {}

    # Segment accuracy
    valid_segments = slices_df.dropna(subset=['gt_segment', 'pred_segment'])
    segment_matches = valid_segments['segment_match']
    metrics['segment_accuracy'] = segment_matches.mean() if not segment_matches.empty else 0

    # Mode accuracy (only for moving segments)
    moving_slices = slices_df[slices_df['gt_segment'] == 'moving']
    valid_modes = moving_slices.dropna(subset=['gt_mode', 'pred_mode'])
    mode_matches = valid_modes['mode_match']
    metrics['mode_accuracy'] = mode_matches.mean() if not mode_matches.empty else 0

    # Purpose accuracy
    valid_purposes = slices_df.dropna(subset=['gt_purpose', 'pred_purpose'])
    purpose_matches = valid_purposes['purpose_match']
    metrics['purpose_accuracy'] = purpose_matches.mean() if not purpose_matches.empty else 0

    # Home/Work purpose accuracy
    if 'gt_purpose' in slices_df.columns:
        home_work_mask = slices_df['gt_purpose'].str.lower().str.contains('home|work', na=False)
        home_work_slices = slices_df[home_work_mask]
        valid_hw = home_work_slices.dropna(subset=['gt_purpose', 'pred_purpose'])
        hw_matches = valid_hw['purpose_match']
        metrics['home_work_purpose_accuracy'] = hw_matches.mean() if not hw_matches.empty else 0

        # Count home and work matches
        if not home_work_slices.empty:
            home_mask = home_work_slices['gt_purpose'].str.lower().str.contains('home', na=False)
            work_mask = home_work_slices['gt_purpose'].str.lower().str.contains('work', na=False)

            home_slices = home_work_slices[home_mask]
            work_slices = home_work_slices[work_mask]

            metrics['home_count'] = len(home_slices)
            metrics['work_count'] = len(work_slices)

            if not home_slices.empty:
                home_correct = home_slices['purpose_match'].sum()
                metrics['home_correct'] = home_correct
            else:
                metrics['home_correct'] = 0

            if not work_slices.empty:
                work_correct = work_slices['purpose_match'].sum()
                metrics['work_correct'] = work_correct
            else:
                metrics['work_correct'] = 0
    else:
        metrics['home_work_purpose_accuracy'] = 0

    # Time coverage
    # In this approach, time coverage is implicitly 100% since we're evaluating all time slices
    metrics['time_coverage'] = 1.0

    # Calculate per-user metrics
    user_metrics = {}

    for email, user_slices in slices_df.groupby('email'):
        user_metrics[email] = {
            'num_slices': len(user_slices)
        }

        # Segment accuracy
        valid_segments = user_slices.dropna(subset=['gt_segment', 'pred_segment'])
        segment_matches = valid_segments['segment_match']
        user_metrics[email]['segment_accuracy'] = segment_matches.mean() if not segment_matches.empty else 0

        # Mode accuracy
        moving_slices = user_slices[user_slices['gt_segment'] == 'moving']
        valid_modes = moving_slices.dropna(subset=['gt_mode', 'pred_mode'])
        mode_matches = valid_modes['mode_match']
        user_metrics[email]['mode_accuracy'] = mode_matches.mean() if not mode_matches.empty else 0

        # Purpose accuracy
        valid_purposes = user_slices.dropna(subset=['gt_purpose', 'pred_purpose'])
        purpose_matches = valid_purposes['purpose_match']
        user_metrics[email]['purpose_accuracy'] = purpose_matches.mean() if not purpose_matches.empty else 0

        # Home/Work purpose accuracy
        if 'gt_purpose' in user_slices.columns:
            home_work_mask = user_slices['gt_purpose'].str.lower().str.contains('home|work', na=False)
            home_work_slices = user_slices[home_work_mask]
            valid_hw = home_work_slices.dropna(subset=['gt_purpose', 'pred_purpose'])
            hw_matches = valid_hw['purpose_match']
            user_metrics[email]['home_work_purpose_accuracy'] = hw_matches.mean() if not hw_matches.empty else 0
        else:
            user_metrics[email]['home_work_purpose_accuracy'] = 0

    metrics['per_user'] = user_metrics

    # Analyze prediction errors (similar to the original approach)
    error_analysis = analyze_timeslice_errors(slices_df)
    metrics['error_analysis'] = error_analysis

    return metrics


def analyze_timeslice_errors(slices_df):
    """
    Analyze patterns in prediction errors based on time slices

    Parameters:
    slices_df (pandas.DataFrame): DataFrame with time slices

    Returns:
    dict: Dictionary with error analysis metrics
    """
    error_analysis = {}

    if slices_df.empty:
        return error_analysis

    # 1. Analyze transition errors (false negatives/positives)
    segment_confusion = pd.crosstab(
        slices_df['gt_segment'],
        slices_df['pred_segment'],
        rownames=['Ground Truth'],
        colnames=['Prediction'],
        dropna=True
    )

    # Convert to dictionary for easy access in reports
    error_analysis['segment_confusion'] = segment_confusion.to_dict()

    # Calculate transition errors
    transition_errors = {
        'missed_slices': 0,  # ground truth = moving, prediction = stationary
        'false_slices': 0  # ground truth = stationary, prediction = moving
    }

    valid_segments = slices_df.dropna(subset=['gt_segment', 'pred_segment'])
    if not valid_segments.empty:
        # Count segment mismatches
        missed_slices = valid_segments[(valid_segments['gt_segment'] == 'moving') &
                                       (valid_segments['pred_segment'] == 'stationary')]
        transition_errors['missed_slices'] = len(missed_slices)

        false_slices = valid_segments[(valid_segments['gt_segment'] == 'stationary') &
                                      (valid_segments['pred_segment'] == 'moving')]
        transition_errors['false_slices'] = len(false_slices)

        # Calculate percentages
        total_segments = len(valid_segments)
        transition_errors['missed_slice_pct'] = transition_errors['missed_slices'] / total_segments * 100
        transition_errors['false_slice_pct'] = transition_errors['false_slices'] / total_segments * 100

    error_analysis['transition_errors'] = transition_errors

    # 2. Analyze mode confusion for moving segments
    moving_slices = slices_df[slices_df['gt_segment'] == 'moving']
    valid_modes = moving_slices.dropna(subset=['gt_mode', 'pred_mode'])

    if not valid_modes.empty:
        mode_confusion = pd.crosstab(
            valid_modes['gt_mode'],
            valid_modes['pred_mode'],
            rownames=['Ground Truth'],
            colnames=['Prediction']
        )

        # Convert to dictionary for reports
        error_analysis['mode_confusion'] = mode_confusion.to_dict()

    # 3. Analyze daily patterns - only if slice_start column exists and is datetime type
    if 'slice_start' in slices_df.columns and pd.api.types.is_datetime64_dtype(slices_df['slice_start']):
        # Create hour column
        slices_df['hour'] = slices_df['slice_start'].dt.hour

        # Segment accuracy by hour
        hourly_segment = slices_df.groupby('hour')['segment_match'].mean().reset_index()
        hourly_segment.columns = ['hour', 'segment_accuracy']

        # Mode accuracy by hour - only if we have moving slices with hour column
        if not moving_slices.empty:
            # Add hour column to moving_slices if it doesn't exist
            if 'hour' not in moving_slices.columns:
                moving_slices = moving_slices.copy()
                moving_slices['hour'] = moving_slices['slice_start'].dt.hour

            # Only compute if we have mode_match column
            if 'mode_match' in moving_slices.columns:
                hourly_mode = moving_slices.groupby('hour')['mode_match'].mean().reset_index()
                hourly_mode.columns = ['hour', 'mode_accuracy']

                # Combine hourly metrics
                hourly_metrics = pd.merge(hourly_segment, hourly_mode, on='hour', how='outer')
            else:
                hourly_metrics = hourly_segment
        else:
            hourly_metrics = hourly_segment

        # Add to error analysis
        error_analysis['hourly_accuracy'] = hourly_metrics.to_dict('records')

    return error_analysis


def generate_timeslice_report(metrics, slices_df, output_dir):
    """
    Generate validation report based on time slice metrics

    Parameters:
    metrics (dict): Dictionary with calculated metrics
    slices_df (pandas.DataFrame): DataFrame with time slices
    output_dir (str): Directory to save report files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Overall metrics report
    overall_report = {
        'Metric': ['Segment Accuracy', 'Mode Accuracy', 'Purpose Accuracy',
                   'Home/Work Purpose Accuracy', 'Time Coverage', 'Total Time Slices'],
        'Value': [
            metrics['segment_accuracy'],
            metrics['mode_accuracy'],
            metrics.get('purpose_accuracy', 0),
            metrics.get('home_work_purpose_accuracy', 0),
            metrics['time_coverage'],
            len(slices_df)
        ]
    }

    # Add home/work specific counts if available
    if 'home_count' in metrics:
        overall_report['Metric'].extend(['Home Count', 'Home Correct', 'Work Count', 'Work Correct'])
        overall_report['Value'].extend([
            metrics.get('home_count', 0),
            metrics.get('home_correct', 0),
            metrics.get('work_count', 0),
            metrics.get('work_correct', 0)
        ])

    overall_df = pd.DataFrame(overall_report)
    overall_df.to_csv(os.path.join(output_dir, 'timeslice_overall_metrics.csv'), index=False)

    # Per-user metrics report
    user_rows = []
    for email, user_metric in metrics['per_user'].items():
        user_rows.append({
            'Email': email,
            'Segment_Accuracy': user_metric['segment_accuracy'],
            'Mode_Accuracy': user_metric['mode_accuracy'],
            'Purpose_Accuracy': user_metric.get('purpose_accuracy', 0),
            'Home_Work_Purpose_Accuracy': user_metric.get('home_work_purpose_accuracy', 0),
            'Num_Slices': user_metric['num_slices']
        })

    user_df = pd.DataFrame(user_rows)
    if not user_df.empty:
        user_df.to_csv(os.path.join(output_dir, 'timeslice_user_metrics.csv'), index=False)

    # Error analysis report
    if 'error_analysis' in metrics:
        error_analysis = metrics['error_analysis']

        # Transition errors
        if 'transition_errors' in error_analysis:
            errors = error_analysis['transition_errors']

            error_df = pd.DataFrame({
                'Error Type': ['Missed Moving Slices', 'False Moving Slices'],
                'Count': [
                    errors['missed_slices'],
                    errors['false_slices']
                ],
                'Percentage': [
                    errors.get('missed_slice_pct', 0),
                    errors.get('false_slice_pct', 0)
                ]
            })
            error_df.to_csv(os.path.join(output_dir, 'timeslice_transition_errors.csv'), index=False)

        # Segment confusion matrix
        if 'segment_confusion' in error_analysis:
            segment_confusion = pd.DataFrame(error_analysis['segment_confusion'])
            segment_confusion.to_csv(os.path.join(output_dir, 'timeslice_segment_confusion.csv'))

        # Mode confusion matrix
        if 'mode_confusion' in error_analysis:
            mode_confusion = pd.DataFrame(error_analysis['mode_confusion'])
            mode_confusion.to_csv(os.path.join(output_dir, 'timeslice_mode_confusion.csv'))

        # Hourly accuracy patterns
        if 'hourly_accuracy' in error_analysis:
            hourly_df = pd.DataFrame(error_analysis['hourly_accuracy'])
            hourly_df.to_csv(os.path.join(output_dir, 'timeslice_hourly_accuracy.csv'), index=False)

    # Save time slices for detailed analysis
    if not slices_df.empty:
        slices_df.to_csv(os.path.join(output_dir, 'detailed_time_slices.csv'), index=False)

    print(f"Time-slice reports saved to {output_dir}")

    # Print a summary
    print("\n===== TIME-SLICE VALIDATION SUMMARY =====")
    print(f"Segment Accuracy: {metrics['segment_accuracy']:.2%}")
    print(f"Mode Accuracy: {metrics['mode_accuracy']:.2%}")
    if 'purpose_accuracy' in metrics:
        print(f"Purpose Accuracy: {metrics['purpose_accuracy']:.2%}")
    if 'home_work_purpose_accuracy' in metrics:
        print(f"Home/Work Purpose Accuracy: {metrics['home_work_purpose_accuracy']:.2%}")
    print(f"Total Time Slices: {len(slices_df)}")
    print(f"Number of Users: {len(metrics['per_user'])}")

    # Print error analysis summary
    if 'error_analysis' in metrics and 'transition_errors' in metrics['error_analysis']:
        errors = metrics['error_analysis']['transition_errors']
        print("\n--- Time Slice Errors ---")
        print(f"Missed Moving Slices: {errors['missed_slices']} ({errors.get('missed_slice_pct', 0):.1f}%)")
        print(f"False Moving Slices: {errors['false_slices']} ({errors.get('false_slice_pct', 0):.1f}%)")

    print("========================================")


def validate_timeslices(ground_truth_df, predictions_df, output_dir, interval_minutes=5):
    """
    Validate using the time-slice method

    Parameters:
    ground_truth_df (pandas.DataFrame): Ground truth DataFrame
    predictions_df (pandas.DataFrame): Predictions DataFrame
    output_dir (str): Directory to save report files
    interval_minutes (int): Size of time intervals in minutes

    Returns:
    tuple: (metrics, slices_df) if successful, None otherwise
    """
    # Create time slices
    print(f"Creating {interval_minutes}-minute time slices...")
    time_slices = create_time_slices(ground_truth_df, predictions_df, interval_minutes)

    # Skip if no time slices
    if not time_slices:
        print("No matching time slices found")
        return None

    # Flatten time slices for analysis
    all_slices = []
    for email in time_slices:
        for day in time_slices[email]:
            all_slices.extend(time_slices[email][day])

    # Convert to DataFrame
    slices_df = pd.DataFrame(all_slices)

    if slices_df.empty:
        print("No valid time slices to analyze")
        return None

    # Calculate metrics
    print("Calculating time-slice metrics...")
    metrics = calculate_timeslice_metrics(time_slices)

    # Generate report
    print("Generating time-slice report...")
    generate_timeslice_report(metrics, slices_df, output_dir)

    return metrics, slices_df


def timeslice_validation(ground_truth_path, predictions_path, output_dir='timeslice_results', interval_minutes=5):
    """
    Main function for time-slice validation

    Parameters:
    ground_truth_path (str): Path to ground truth file or folder
    predictions_path (str): Path to predictions file or folder
    output_dir (str): Directory to save report files
    interval_minutes (int): Size of time intervals in minutes

    Returns:
    tuple: (metrics, slices_df) if successful, None otherwise
    """
    # Import required functions from main validation script
    from PythiaValidation import (
        parse_ground_truth_data, load_prediction_data,
        normalize_dates, normalize_segment_values, normalize_purpose_values,
        normalize_mode_values, normalize_durations,
        scan_dataset_folder, combine_datasets
    )

    # Check if inputs are directories or files
    if os.path.isfile(ground_truth_path) and os.path.isfile(predictions_path):
        print("Processing single files mode")

        # Load data
        print(f"Loading ground truth data from: {ground_truth_path}")
        ground_truth_df = parse_ground_truth_data(ground_truth_path)

        print(f"Loading prediction data from: {predictions_path}")
        predictions_df = load_prediction_data(predictions_path)

        if ground_truth_df.empty:
            print("Error: Ground truth dataset is empty")
            return None

        if predictions_df.empty:
            print("Error: Predictions dataset is empty")
            return None

        # Normalize data
        print("Normalizing data...")
        ground_truth_df = normalize_dates(ground_truth_df, ['startTime', 'endTime'], adjust_timezone=False)
        predictions_df = normalize_dates(predictions_df, ['startTime', 'endTime'], adjust_timezone=True)

        ground_truth_df = normalize_segment_values(ground_truth_df, 'segment')
        predictions_df = normalize_segment_values(predictions_df, 'segment')

        if 'purpose' in ground_truth_df.columns:
            ground_truth_df = normalize_purpose_values(ground_truth_df, 'purpose')
        if 'purpose' in predictions_df.columns:
            predictions_df = normalize_purpose_values(predictions_df, 'purpose')

        ground_truth_df, predictions_df = normalize_mode_values(ground_truth_df, predictions_df)
        ground_truth_df, predictions_df = normalize_durations(ground_truth_df, predictions_df)

        # Run time-slice validation
        return validate_timeslices(ground_truth_df, predictions_df, output_dir, interval_minutes)

    # Directory processing
    if not os.path.isdir(ground_truth_path):
        print(f"Error: Ground truth path '{ground_truth_path}' is not a directory")
        return None

    if not os.path.isdir(predictions_path):
        print(f"Error: Predictions path '{predictions_path}' is not a directory")
        return None

    # Scan folders for files by user
    print(f"Scanning ground truth folder: {ground_truth_path}")
    ground_truth_files = scan_dataset_folder(ground_truth_path)

    print(f"Scanning predictions folder: {predictions_path}")
    prediction_files = scan_dataset_folder(predictions_path)

    # Find users present in both datasets
    common_users = set(ground_truth_files.keys()) & set(prediction_files.keys())
    print(f"Found {len(common_users)} users with data in both folders")

    # If no common users, try matching by filename
    if not common_users:
        print("No common users found by email, trying to match by filename...")

        # Get base filenames without extensions
        gt_basenames = {os.path.splitext(os.path.basename(f))[0]: f
                        for user_files in ground_truth_files.values()
                        for f in user_files}

        pred_basenames = {os.path.splitext(os.path.basename(f))[0]: f
                          for user_files in prediction_files.values()
                          for f in user_files}

        common_basenames = set(gt_basenames.keys()) & set(pred_basenames.keys())

        if common_basenames:
            print(f"Found {len(common_basenames)} common filenames")

            # Process each file pair separately and combine results
            all_slices = []

            for name in common_basenames:
                gt_file = gt_basenames[name]
                pred_file = pred_basenames[name]

                print(f"\nProcessing file pair: {name}")

                # Load data
                ground_truth_df = parse_ground_truth_data(gt_file)
                predictions_df = load_prediction_data(pred_file)

                if ground_truth_df.empty or predictions_df.empty:
                    print(f"Skipping file pair {name} due to empty dataset")
                    continue

                # Normalize data
                ground_truth_df = normalize_dates(ground_truth_df, ['startTime', 'endTime'], adjust_timezone=False)
                predictions_df = normalize_dates(predictions_df, ['startTime', 'endTime'], adjust_timezone=True)

                ground_truth_df = normalize_segment_values(ground_truth_df, 'segment')
                predictions_df = normalize_segment_values(predictions_df, 'segment')

                if 'purpose' in ground_truth_df.columns:
                    ground_truth_df = normalize_purpose_values(ground_truth_df, 'purpose')
                if 'purpose' in predictions_df.columns:
                    predictions_df = normalize_purpose_values(predictions_df, 'purpose')

                ground_truth_df, predictions_df = normalize_mode_values(ground_truth_df, predictions_df)
                ground_truth_df, predictions_df = normalize_durations(ground_truth_df, predictions_df)

                # Create time slices
                time_slices = create_time_slices(ground_truth_df, predictions_df, interval_minutes)

                # Add slices to all_slices
                for email in time_slices:
                    for day in time_slices[email]:
                        all_slices.extend(time_slices[email][day])

            if all_slices:
                # Convert to DataFrame
                slices_df = pd.DataFrame(all_slices)

                # Calculate metrics
                print("\nCalculating overall time-slice metrics...")
                metrics = calculate_timeslice_metrics({'combined': {'all': all_slices}})

                # Generate report
                print("Generating time-slice report...")
                generate_timeslice_report(metrics, slices_df, output_dir)

                return metrics, slices_df
            else:
                print("No valid time slices found across file pairs")
                return None

    # Process by user
    all_slices = []
    user_metrics = {}

    for user in common_users:
        print(f"\nProcessing user: {user}")

        # Combine ground truth files for this user
        print(f"Loading ground truth data from {len(ground_truth_files[user])} files")
        ground_truth_df = combine_datasets(ground_truth_files[user], parse_ground_truth_data)

        # Combine prediction files for this user
        print(f"Loading prediction data from {len(prediction_files[user])} files")
        predictions_df = combine_datasets(prediction_files[user], load_prediction_data)

        if ground_truth_df.empty:
            print(f"Warning: Ground truth dataset is empty for user {user}")
            continue

        if predictions_df.empty:
            print(f"Warning: Predictions dataset is empty for user {user}")
            continue

        # Normalize data
        print("Normalizing data...")
        ground_truth_df = normalize_dates(ground_truth_df, ['startTime', 'endTime'], adjust_timezone=False)
        predictions_df = normalize_dates(predictions_df, ['startTime', 'endTime'], adjust_timezone=True)

        ground_truth_df = normalize_segment_values(ground_truth_df, 'segment')
        predictions_df = normalize_segment_values(predictions_df, 'segment')

        if 'purpose' in ground_truth_df.columns:
            ground_truth_df = normalize_purpose_values(ground_truth_df, 'purpose')
        if 'purpose' in predictions_df.columns:
            predictions_df = normalize_purpose_values(predictions_df, 'purpose')

        ground_truth_df, predictions_df = normalize_mode_values(ground_truth_df, predictions_df)
        ground_truth_df, predictions_df = normalize_durations(ground_truth_df, predictions_df)

        # Create time slices
        print(f"Creating {interval_minutes}-minute time slices for user {user}...")
        time_slices = create_time_slices(ground_truth_df, predictions_df, interval_minutes)

        if not time_slices or user not in time_slices or not time_slices[user]:
            print(f"No matching time slices found for user {user}")
            continue

        # Flatten user's time slices
        user_slices = []
        for day in time_slices[user]:
            user_slices.extend(time_slices[user][day])

        # Calculate metrics for this user
        print(f"Calculating time-slice metrics for user {user}")
        user_metric = calculate_timeslice_metrics({user: time_slices[user]})
        user_metrics[user] = user_metric

        # Generate user-specific report
        user_output_dir = os.path.join(output_dir, user.split('@')[0])
        os.makedirs(user_output_dir, exist_ok=True)

        user_slices_df = pd.DataFrame(user_slices)
        print(f"Generating time-slice report for user {user}")
        generate_timeslice_report(user_metric, user_slices_df, user_output_dir)

        # Add user slices to all slices
        all_slices.extend(user_slices)

    # Combine all slices and calculate overall metrics
    if all_slices:
        all_slices_df = pd.DataFrame(all_slices)
        print("\nCalculating overall time-slice metrics...")
        overall_metrics = calculate_timeslice_metrics({'combined': {'all': all_slices}})

        # Add per-user metrics
        overall_metrics['per_user'] = {u: m['per_user'].get(u, {}) for u, m in user_metrics.items()}

        # Generate overall report
        print("Generating overall time-slice report...")
        generate_timeslice_report(overall_metrics, all_slices_df, output_dir)

        return overall_metrics, all_slices_df
    else:
        print("No matching time slices found across all users")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Validate travel diary predictions using time-slice method')
    parser.add_argument('--ground-truth', required=True, help='Path to ground truth file or folder')
    parser.add_argument('--predictions', required=True, help='Path to predictions file or folder')
    parser.add_argument('--output', default='timeslice_results', help='Output directory for validation results')
    parser.add_argument('--interval', type=int, default=5, help='Time slice interval in minutes')

    args = parser.parse_args()

    timeslice_validation(args.ground_truth, args.predictions, args.output, args.interval)