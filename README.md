# Travel Diary Validation System

## Overview

This system validates predictions from a mobile travel diary app against ground truth data. It analyzes how well the app predicts user activities, including segment types (stationary/moving), transportation modes, and trip purposes.

The validation system offers two complementary validation methods:

1. **Entry-Based Validation**: Matches overlapping entries between ground truth and predictions
2. **Time-Slice Validation**: Evaluates prediction accuracy at fixed time intervals (5-minute slices by default)

## Features

- Handles various date formats and data inconsistencies
- Normalizes transportation modes across datasets
- Special focus on home and work purpose detection
- Analyzes patterns in prediction errors (e.g., delayed trip detection)
- Generates detailed validation reports and metrics
- Supports per-user analysis
- Works with both individual files and folders of data

## Requirements

- Python 3.6+
- pandas
- numpy

## Installation

1. Clone the repository
2. Install required packages:

```bash
pip install pandas numpy
```

## File Structure

- `PythiaValidation.py` - Main entry-based validation script
- `TimeSlice.py` - Time-slice validation script
- `RunValidation.py` - Example script to run validation with method selection

## Usage

### Command Line Interface

Run the example script with method selection:

```bash
python RunValidation.py --method [entry|timeslice] --ground-truth <path> --predictions <path> --output <dir> --interval <minutes>
```

Parameters:
- `--method`: Validation method ('entry' or 'timeslice')
- `--ground-truth`: Path to the ground truth folder or file
- `--predictions`: Path to the predictions folder or file
- `--output`: Directory where validation results will be saved
- `--interval`: Time slice interval in minutes (for time-slice method only, default: 5)

### Examples

#### Entry-Based Validation (Default):
```bash
python RunValidation.py --method entry --ground-truth GroundTruth --predictions Predictions
```

#### Time-Slice Validation:
```bash
python RunValidation.py --method timeslice --ground-truth GroundTruth --predictions Predictions --interval 5
```

### Using as a Module

You can also import and use the validation functions in your own Python code:

```python
# Entry-based validation
from PythiaValidation import validate_travel_diaries
results = validate_travel_diaries('GroundTruth', 'Predictions', 'results')

# Time-slice validation
from TimeSlice import timeslice_validation
results = timeslice_validation('GroundTruth', 'Predictions', 'results', interval_minutes=5)
```

## Input Data Format

### Ground Truth Data
The system is flexible and can handle various formats, but ideally should contain:
- Email identifier
- Start time
- End time
- Segment type (stationary/moving)
- Purpose (optional)
- Transportation mode (for moving segments)

### Prediction Data
Should contain:
- Email identifier
- Prediction ID
- Start time
- End time
- Segment type
- Transportation mode (either as JSON or simple string)
- Other optional fields (purpose, location, etc.)

## Validation Methods Explained

### Entry-Based Validation

This method matches entries between ground truth and prediction datasets based on overlapping time periods:

- **Approach**: Finds entries in both datasets that overlap in time for the same user
- **Matching**: Requires at least 10% time overlap to consider entries as matching
- **Flexibility**: Includes a 30-minute tolerance window to account for minor timing differences
- **Output**: Reports what percentage of ground truth entries were correctly identified
- **Best for**: Evaluating how well specific activities were captured and classified

### Time-Slice Validation

This method divides each day into fixed time intervals and compares ground truth with predictions at each interval:

- **Approach**: Creates a complete timeline for the day divided into 5-minute slices (configurable)
- **Evaluation**: For each slice, identifies what's happening in both ground truth and predictions
- **Equal Weight**: Each slice has equal importance, so longer activities count proportionally
- **Output**: Reports what percentage of time slices were correctly identified
- **Best for**: Evaluating continuous timeline accuracy and finding patterns in daily activities

## Output Reports

Both methods generate several CSV files in the output directory:

### Common Reports
- `*_overall_metrics.csv` - Summary of validation metrics
- `*_user_metrics.csv` - Per-user validation metrics
- `*_segment_confusion.csv` - Confusion matrix for segment classification
- `*_mode_confusion.csv` - Confusion matrix for transportation mode classification

### Entry-Based Method Reports
- `trip_start_delay_analysis.csv` - Analysis of trip start detection delays
- `transition_errors.csv` - Analysis of missed and false trips
- `detailed_matches.csv` - Detailed information about matched entries
- `simple_matches.csv` - Simplified view of matches for easier analysis

### Time-Slice Method Reports
- `timeslice_transition_errors.csv` - Analysis of missed and false moving slices
- `timeslice_hourly_accuracy.csv` - Accuracy analysis by hour of day
- `detailed_time_slices.csv` - Detailed information about all time slices

## Metrics

Both validation methods calculate the following metrics:

- **Segment Accuracy**: Percentage of correctly identified segment types (stationary/moving)
- **Mode Accuracy**: Percentage of correctly identified transportation modes (for moving segments)
- **Purpose Accuracy**: Percentage of correctly identified purposes (when available)
- **Home/Work Purpose Accuracy**: Specialized metric for home and work detection
- **Time Coverage**: Percentage of ground truth time that is covered by valid predictions

## Home and Work Purpose Analysis

The system includes specialized purpose accuracy analysis for home and work locations:

- **Purpose Normalization**: Standardizes various home/work-related terms
- **Targeted Metrics**: Calculates dedicated metrics for home and work purpose detection
- **Detailed Reporting**: Shows counts and correct matches for both home and work separately

## Error Pattern Analysis

The system also analyzes patterns in prediction failures:

- **Trip Start Delay Analysis**: Calculates how late the app detects the start of trips
- **Delay Distribution**: Categorizes delays into time buckets (under 1min, 1-5min, 5-15min, over 15min)
- **Error Classification**: Identifies missed trips, false trips, and delayed detections
- **Hourly Analysis**: Shows prediction accuracy by hour of day (time-slice method only)

## Troubleshooting

If you encounter errors:
- **KeyError 'hour'**: Update the analyze_timeslice_errors function to create the hour column before grouping
- **Date parsing issues**: Add your specific date format to the date_formats list
- **No matching entries**: Check if date ranges overlap between ground truth and predictions
- **CSV parsing errors**: Try different delimiter options or manually specify column names

## How to Fix Common Issues

### Time-Slice Hour Error
For the "KeyError: 'hour'" error, ensure the hour column is created before grouping:

```python
# Make sure we create the hour column first
if 'slice_start' in slices_df.columns and pd.api.types.is_datetime64_dtype(slices_df['slice_start']):
    slices_df['hour'] = slices_df['slice_start'].dt.hour
    
    # Also add the hour column to moving_slices
    if not moving_slices.empty:
        moving_slices = moving_slices.copy()
        moving_slices['hour'] = moving_slices['slice_start'].dt.hour
```

### Date Parsing Problems
If your dates aren't being recognized correctly:

1. Check the format of dates in your data
2. Add your specific format to the date_formats list in normalize_dates function
3. Look for possible day/month swaps causing incorrect year detection
