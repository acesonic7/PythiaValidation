#!/usr/bin/env python3
"""
Travel Diary Validation Script (Fixed Version)

This script validates travel diary predictions against ground truth data.
It analyzes how well a mobility tracking app predicts user movements, transportation modes, and visit purposes.

Key fixes in this version:
1. Updated pandas parameter names for compatibility with newer versions
2. Improved date parsing to handle various formats correctly
3. Added date format detection and correction for month/day swap issues
4. Enhanced debug information for troubleshooting date range mismatches
"""

import os
import re
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def scan_dataset_folder(folder_path):
    """
    Scan a folder for CSV dataset files
    """
    user_files = {}

    try:
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv') or filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)

                # Try to determine the user from the filename or by peeking at the content
                user_email = None

                # First try to extract from filename (if it contains an email pattern)
                email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
                email_match = re.search(email_pattern, filename)
                if email_match:
                    user_email = email_match.group(0)

                # If not found in filename, try to peek at the first few lines
                if not user_email:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            head = ''.join([f.readline() for _ in range(5)])
                            email_match = re.search(email_pattern, head)
                            if email_match:
                                user_email = email_match.group(0)
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")

                # If we found a user email, add the file to their list
                if user_email:
                    if user_email not in user_files:
                        user_files[user_email] = []
                    user_files[user_email].append(file_path)
                else:
                    # If no email found, add to a generic "unknown" category
                    if 'unknown' not in user_files:
                        user_files['unknown'] = []
                    user_files['unknown'].append(file_path)
    except Exception as e:
        print(f"Error scanning folder {folder_path}: {e}")

    return user_files


def parse_ground_truth_data(file_path):
    """
    Parse ground truth data from a CSV file or text file
    """
    # First try standard flexible CSV loading
    try:
        # Try to determine the delimiter from the file
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            sample = ''.join([f.readline() for _ in range(5)])

        # Count potential delimiters
        comma_count = sample.count(',')
        tab_count = sample.count('\t')
        semicolon_count = sample.count(';')

        # Choose the most likely delimiter
        if tab_count > comma_count and tab_count > semicolon_count:
            delimiter = '\t'
        elif semicolon_count > comma_count:
            delimiter = ';'
        else:
            delimiter = ','

        df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8', on_bad_lines='skip')
        print(f"Ground truth file columns: {df.columns.tolist()}")

        # Check if expected columns exist or can be created
        if 'email' in df.columns and 'startTime' in df.columns:
            return df

        # Try to extract email from a combined column if it exists
        if 'user' in df.columns and 'email' not in df.columns:
            # See if the email is embedded in the user column
            email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            df['email'] = df['user'].astype(str).str.extract(f'({email_pattern})')[0]

            # If still no emails found, use the user column as an identifier
            if df['email'].isna().all():
                df['email'] = df['user']

            return df
    except Exception as e:
        print(f"Standard CSV parsing failed: {e}")

    # If standard loading fails, try to parse the ground truth manually
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        # Split by lines
        lines = content.strip().split('\n')

        # Initialize lists for each column
        users = []
        emails = []
        start_times = []
        end_times = []
        segments = []
        purposes = []
        durations = []
        modes = []

        # Extract header line if present
        header_line = lines[0] if lines else ""
        header_parts = re.split(r'\s+', header_line.strip())
        has_header = 'user' in header_parts or 'email' in header_parts or 'startTime' in header_parts

        # Process data lines
        for i, line in enumerate(lines):
            # Skip header if it exists
            if i == 0 and has_header:
                continue

            # Skip empty lines
            if not line.strip():
                continue

            # Try to extract structured data
            # First look for email pattern
            email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', line)
            email = email_match.group(0) if email_match else None

            # If no email found, skip this line
            if not email:
                continue

            # Extract other parts based on patterns
            # Pattern for dates: dd/mm/yy HH:MM or dd/mm/yyyy HH:MM
            date_matches = re.findall(r'\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}', line)

            start_time = date_matches[0] if len(date_matches) > 0 else None
            end_time = date_matches[1] if len(date_matches) > 1 else None

            # Extract segment type (Stationary/Moving)
            segment = None
            if 'stationary' in line.lower():
                segment = 'stationary'
            elif 'moving' in line.lower():
                segment = 'moving'

            # Extract purpose if present (usually after Stationary)
            purpose = None
            if segment == 'stationary':
                purpose_match = re.search(r'stationary\s+([a-zA-Z()]+)', line, re.IGNORECASE)
                if purpose_match:
                    purpose = purpose_match.group(1)

            # Extract duration and mode
            duration_match = re.search(r'\b(\d+)\s*$', line)
            duration = duration_match.group(1) if duration_match else None

            mode_patterns = ['cycling', 'walking', 'car', 'car+walking']
            mode = None
            for pattern in mode_patterns:
                if pattern in line.lower():
                    mode = pattern
                    break

            # Append to lists
            users.append(None)  # No separate user field
            emails.append(email)
            start_times.append(start_time)
            end_times.append(end_time)
            segments.append(segment)
            purposes.append(purpose)
            durations.append(duration)
            modes.append(mode)

        # Create DataFrame
        data = {
            'email': emails,
            'startTime': start_times,
            'endTime': end_times,
            'segment': segments,
            'purpose': purposes,
            'duration': durations,
            'mode': modes
        }

        if any(emails):
            return pd.DataFrame(data)
        else:
            print("Could not extract any structured data from the file")
            return pd.DataFrame()
    except Exception as e:
        print(f"Manual parsing failed: {e}")
        return pd.DataFrame()


def load_prediction_data(file_path):
    """
    Load prediction data from CSV file with flexible parsing
    """
    try:
        # First attempt: parse with different delimiter options
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            sample = ''.join([f.readline() for _ in range(5)])

        # Count potential delimiters
        comma_count = sample.count(',')
        tab_count = sample.count('\t')
        semicolon_count = sample.count(';')

        # Choose the most likely delimiter
        if tab_count > comma_count and tab_count > semicolon_count:
            delimiter = '\t'
        elif semicolon_count > comma_count:
            delimiter = ';'
        else:
            delimiter = ','

        # Try parsing with the chosen delimiter
        df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8', on_bad_lines='skip')
        print(f"Prediction file columns: {df.columns.tolist()}")

        # Check for required columns and rename if needed
        if 'email' not in df.columns:
            # Look for email column with different names
            for col in df.columns:
                if 'email' in col.lower() or 'mail' in col.lower():
                    df['email'] = df[col]
                    break

            # If still not found, try to extract from other columns
            if 'email' not in df.columns:
                for col in df.columns:
                    if df[col].dtype == 'object':  # Only check string columns
                        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
                        emails = df[col].astype(str).str.extract(f'({email_pattern})')[0]
                        if emails.notna().any():
                            df['email'] = emails
                            break

        # Add missing required columns if they don't exist
        if 'startTime' not in df.columns:
            for col in df.columns:
                if 'start' in col.lower() or 'time' in col.lower():
                    df['startTime'] = df[col]
                    break

        if 'endTime' not in df.columns:
            for col in df.columns:
                if 'end' in col.lower() or 'finish' in col.lower():
                    df['endTime'] = df[col]
                    break

        if 'segment' not in df.columns:
            for col in df.columns:
                if 'segment' in col.lower() or 'activity' in col.lower():
                    df['segment'] = df[col]
                    break

        # If we got this far, we have a dataframe to work with
        return df
    except Exception as e:
        print(f"First parsing attempt failed: {e}")

    # Second attempt: try with pandas flexible parser
    try:
        # Use more permissive parsing settings
        df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8',
                         on_bad_lines='skip', dtype=str)
        print(f"Second attempt successful, columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Second parsing attempt failed: {e}")

    # Third attempt: try with manual line-by-line parsing
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.readlines()

        # Try to identify delimiter from first few lines
        potential_delimiters = [',', '\t', ';', '|']
        delimiter_counts = {d: 0 for d in potential_delimiters}

        for line in content[:5]:
            for d in potential_delimiters:
                delimiter_counts[d] += line.count(d)

        best_delimiter = max(delimiter_counts.items(), key=lambda x: x[1])[0]

        # Parse lines
        rows = []
        headers = None

        for i, line in enumerate(content):
            if not line.strip():
                continue

            fields = line.strip().split(best_delimiter)

            if i == 0:
                headers = [f.strip() for f in fields]

                # Make sure we have required columns
                if 'email' not in headers:
                    for j, h in enumerate(headers):
                        if 'mail' in h.lower():
                            headers[j] = 'email'
                            break

                if 'startTime' not in headers:
                    for j, h in enumerate(headers):
                        if 'start' in h.lower():
                            headers[j] = 'startTime'
                            break

                if 'endTime' not in headers:
                    for j, h in enumerate(headers):
                        if 'end' in h.lower():
                            headers[j] = 'endTime'
                            break

                if 'segment' not in headers:
                    for j, h in enumerate(headers):
                        if 'segment' in h.lower() or 'activity' in h.lower():
                            headers[j] = 'segment'
                            break
            else:
                # Make sure we have enough fields
                while len(fields) < len(headers):
                    fields.append(None)

                # Trim fields if we have too many
                fields = fields[:len(headers)]

                rows.append(dict(zip(headers, fields)))

        df = pd.DataFrame(rows)
        print(f"Manual parsing successful, columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Third parsing attempt failed: {e}")

    # All parsing attempts failed
    print(f"All parsing attempts failed for {file_path}")
    return pd.DataFrame()


def combine_datasets(file_paths, parser_func):
    """
    Combine multiple datasets from the same user
    """
    combined_df = pd.DataFrame()

    for file_path in file_paths:
        try:
            df = parser_func(file_path)
            if not df.empty:
                combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    return combined_df


def normalize_dates(df, date_columns, adjust_timezone=False):
    """
    Normalize date formats to datetime objects
    """
    for col in date_columns:
        if col in df.columns:
            # Convert to string first to handle various input types
            df[col] = df[col].astype(str)

            # Handle 'NULL' values
            df[col] = df[col].replace('NULL', pd.NA).replace('null', pd.NA).replace('None', pd.NA).replace('nan', pd.NA)

            # Try multiple date formats
            date_formats = [
                '%d/%m/%y %H:%M',  # 11/3/25 8:21
                '%d/%m/%Y %H:%M',  # 11/3/2025 8:21
                '%Y-%m-%d %H:%M:%S.%f',  # 2025-03-23 22:25:20.933000
                '%d/%m/%y %H:%M:%S.%f',  # 25/3/25 15:44
                '%d/%m/%Y %H:%M:%S.%f',  # 25/3/2025 15:44
                '%Y-%m-%d %H:%M:%S',  # 2025-03-23 22:25:20
                '%d/%m/%Y',  # 25/3/2025
                '%d-%m-%Y %H:%M',  # 25-3-2025 15:44
                '%d-%m-%Y',  # 25-3-2025
                '%Y/%m/%d %H:%M:%S',  # 2025/03/23 22:25:20
                '%m/%d/%Y %H:%M:%S',  # 9/3/2025 14:29:00 (US format)
                '%m/%d/%y %H:%M:%S',  # 9/3/25 14:29:00 (US format)
                '%m/%d/%Y %H:%M',  # 9/3/2025 14:29 (US format)
                '%m/%d/%y %H:%M',  # 9/3/25 14:29 (US format)
                '%Y-%m-%d',  # 2025-03-23
            ]

            success = False
            for date_format in date_formats:
                try:
                    temp_dates = pd.to_datetime(df[col], format=date_format, errors='coerce')
                    # If enough dates were parsed successfully, use this format
                    if temp_dates.notna().mean() > 0.5:
                        df[col] = temp_dates
                        print(f"Successfully parsed dates in {col} using format: {date_format}")
                        success = True
                        break
                except:
                    continue

            # Final attempt with pandas default parser
            if not success or df[col].isna().all() or not pd.api.types.is_datetime64_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if df[col].notna().any():
                        print(f"Parsed dates in {col} using pandas default parser")
                except:
                    print(f"Failed to parse dates in column {col}")

            # Check for and fix potential date format issues (month/day swap)
            # Use 2025 as the current year (or adjust as needed)
            current_year = 2025
            if pd.api.types.is_datetime64_dtype(df[col]) and df[col].notna().any():
                # Get distribution of years for diagnostic purposes
                year_counts = df[col].dt.year.value_counts().sort_index()
                print(f"Years in {col}: {year_counts.to_dict()}")

                # Check for dates in distant future
                future_mask = df[col].dt.year > current_year + 1
                if future_mask.any():
                    future_count = future_mask.sum()
                    print(
                        f"Warning: Found {future_count} dates in {col} that appear to be in the future (> {current_year + 1})")
                    print("This may indicate a date format issue (month/day vs. day/month)")

                    # Try to fix by swapping day and month for these dates
                    problem_dates = df.loc[future_mask, col].copy()
                    for idx in problem_dates.index:
                        date_str = df.loc[idx, col].strftime('%Y-%m-%d %H:%M:%S')
                        year = int(date_str.split('-')[0])
                        if year > current_year + 1:
                            try:
                                # Extract parts
                                parts = date_str.split('-')
                                month = parts[1]
                                day_part = parts[2].split(' ')
                                day = day_part[0]
                                time_part = day_part[1] if len(day_part) > 1 else "00:00:00"

                                # Only swap if both could be valid month/day
                                if 1 <= int(month) <= 12 and 1 <= int(day) <= 12:
                                    # Create new date with swapped month/day
                                    new_date_str = f"{parts[0]}-{day}-{month} {time_part}"
                                    try:
                                        new_date = pd.to_datetime(new_date_str)
                                        df.loc[idx, col] = new_date
                                        print(f"Fixed date by swapping month/day: {date_str} -> {new_date_str}")
                                    except:
                                        print(f"Failed to create valid date from: {new_date_str}")
                            except Exception as e:
                                print(f"Error fixing date {date_str}: {e}")

            # Adjust timezone by adding 1 hour if requested
            if adjust_timezone and df[col].notna().any():
                df[col] = df[col] + pd.Timedelta(hours=1)
                print(f"Applied timezone adjustment (+1 hour) to {col}")

    return df


def normalize_segment_values(df, segment_col):
    """
    Normalize segment values (stationary/moving) to lowercase
    """
    if segment_col in df.columns and df[segment_col].notna().any():
        # Convert to lowercase
        df[segment_col] = df[segment_col].astype(str).str.lower()

        # Replace common variations
        replacements = {
            'stat': 'stationary',
            'still': 'stationary',
            'move': 'moving',
            'mov': 'moving',
            'motion': 'moving'
        }

        for old, new in replacements.items():
            df[segment_col] = df[segment_col].str.replace(old, new, regex=False)

        # Normalize to just stationary/moving
        df[segment_col] = df[segment_col].apply(
            lambda x: 'stationary' if 'station' in str(x).lower() else
            ('moving' if 'mov' in str(x).lower() else x)
        )

    return df


def normalize_purpose_values(df, purpose_col):
    if purpose_col in df.columns and df[purpose_col].notna().any():
        # Create normalized version
        df['normalized_purpose'] = df[purpose_col].astype(str).str.lower().str.strip()

        # Apply exact standardization (not pattern matching)
        home_patterns = ['home', 'house', 'apartment', 'residence', 'flat']
        work_patterns = ['work', 'office', 'job', 'workplace', 'company']

        for pattern in home_patterns:
            mask = df['normalized_purpose'].str.contains(pattern, case=False, na=False)
            df.loc[mask, 'normalized_purpose'] = 'home'  # Exactly 'home'

        for pattern in work_patterns:
            mask = df['normalized_purpose'].str.contains(pattern, case=False, na=False)
            df.loc[mask, 'normalized_purpose'] = 'work'  # Exactly 'work'

    return df


def extract_mode_from_json(mode_json):
    """
    Extract the dominant transportation mode from JSON string
    """
    if pd.isna(mode_json) or not mode_json:
        return None

    try:
        # If it's already a string mode, return it directly
        if isinstance(mode_json, str) and not mode_json.startswith('{'):
            return mode_json.strip().lower()

        # Convert single quotes to double quotes for valid JSON
        if isinstance(mode_json, str):
            mode_json = mode_json.replace("'", '"')

            # Handle JSON strings like {key:value} without quotes around keys
            # This is a common format issue
            mode_json = re.sub(r'{\s*([^"\'{\s][^:]*?):', r'{\"\1\":', mode_json)
            mode_json = re.sub(r',\s*([^"\'{\s][^:]*?):', r',\"\1\":', mode_json)

        # Try to parse as JSON
        if isinstance(mode_json, str) and (mode_json.startswith('{') or mode_json.startswith('[')):
            try:
                mode_dict = json.loads(mode_json)
            except json.JSONDecodeError:
                # If standard JSON parsing fails, try a more forgiving approach
                # Sometimes values can be unquoted, so use ast.literal_eval
                import ast
                mode_dict = ast.literal_eval(mode_json)
        else:
            # Not JSON formatted
            return str(mode_json).lower()

        # If it's a dictionary, find the mode with the highest percentage
        if isinstance(mode_dict, dict) and mode_dict:
            # Convert all values to float for comparison
            mode_dict = {
                k: float(v) if (isinstance(v, (int, float, str)) and str(v).replace('.', '', 1).isdigit()) else 0
                for k, v in mode_dict.items()}

            # Check if dictionary is empty after conversion
            if not mode_dict:
                return None

            dominant_mode = max(mode_dict.items(), key=lambda x: x[1])
            return dominant_mode[0] if float(dominant_mode[1]) > 0 else None
        elif isinstance(mode_dict, list) and mode_dict:
            # If it's a list, return the first value
            return str(mode_dict[0]).lower()
        else:
            return None
    except Exception as e:
        print(f"Error parsing mode JSON: {e}, Value: {mode_json}")
        # Try to extract any mode-like string from the JSON
        if isinstance(mode_json, str):
            mode_patterns = ['walking', 'running', 'cycling', 'motorcycle', 'in_vehicle', 'on_bicycle', 'driving', 'car']
            for pattern in mode_patterns:
                if pattern in mode_json.lower():
                    return pattern
        return None


def normalize_mode_values(ground_truth_df, predictions_df):
    """
    Normalize transportation mode values with strict debugging
    """
    print("=== MODE NORMALIZATION DEBUGGING ===")

    # Step 1: Create a simple, standardized mapping (with minimal transformations)
    mode_mapping = {
        # Keep original values where possible
        'motorcycle': 'motorcycle',
        'on_bicycle': 'on_bicycle',
        'bicycle': 'on_bicycle',
        'cycling': 'on_bicycle',
        'cycle': 'on_bicycle',
        'bike': 'on_bicycle',
        'walking': 'walking',
        'walk': 'walking',
        'in_vehicle': 'in_vehicle',
        'car': 'in_vehicle',
        'vehicle': 'in_vehicle',
        'running': 'running',
        'unknown': 'unknown',
        'still': 'stationary'
    }

    # Step 2: Add direct debugging for ground truth
    if 'mode' in ground_truth_df.columns:
        print(f"Ground truth unique modes BEFORE normalization: {ground_truth_df['mode'].unique()}")
        ground_truth_df['normalized_mode'] = ground_truth_df['mode'].copy()

        # Apply mapping directly (no string manipulation)
        for i, row in ground_truth_df.iterrows():
            mode = str(row['mode']).lower() if not pd.isna(row['mode']) else None
            if mode in mode_mapping:
                ground_truth_df.at[i, 'normalized_mode'] = mode_mapping[mode]

        print(f"Ground truth unique modes AFTER normalization: {ground_truth_df['normalized_mode'].unique()}")

    # Step 3: Add direct debugging for predictions
    if 'smode' in predictions_df.columns:
        print(f"Prediction unique modes BEFORE normalization: {predictions_df['smode'].unique()}")
        predictions_df['normalized_mode'] = predictions_df['smode'].copy()

        # Apply mapping directly (no string manipulation)
        for i, row in predictions_df.iterrows():
            smode = str(row['smode']).lower() if not pd.isna(row['smode']) else None
            if smode in mode_mapping:
                predictions_df.at[i, 'normalized_mode'] = mode_mapping[smode]

        print(f"Prediction unique modes AFTER normalization: {predictions_df['normalized_mode'].unique()}")

    return ground_truth_df, predictions_df


def normalize_durations(ground_truth_df, predictions_df):
    """
    Normalize duration values (minutes in ground truth, seconds in predictions)
    """
    # Convert ground truth duration from minutes to seconds
    if 'duration' in ground_truth_df.columns:
        # Ensure it's numeric
        ground_truth_df['duration'] = pd.to_numeric(ground_truth_df['duration'], errors='coerce')
        # Convert to seconds
        ground_truth_df['duration_seconds'] = ground_truth_df['duration'] * 60

    # Ensure prediction duration is in seconds
    if 'duration' in predictions_df.columns:
        predictions_df['duration_seconds'] = pd.to_numeric(predictions_df['duration'], errors='coerce')

    return ground_truth_df, predictions_df


def find_overlapping_entries(ground_truth_df, predictions_df):
    """
    Find entries in both datasets that overlap in time for the same user
    """
    matches = []

    # Print time ranges for debugging
    if not ground_truth_df.empty and 'startTime' in ground_truth_df.columns and ground_truth_df[
        'startTime'].notna().any():
        gt_min = ground_truth_df['startTime'].min()
        gt_max = ground_truth_df['startTime'].max()
        print(f"Ground truth date range: {gt_min} to {gt_max}")

        # Add year distribution for better diagnostics
        year_counts = ground_truth_df['startTime'].dt.year.value_counts().sort_index()
        print(f"Ground truth years distribution: {year_counts.to_dict()}")

        # Show month distribution for the most common year
        if not year_counts.empty:
            most_common_year = year_counts.idxmax()
            month_counts = ground_truth_df[ground_truth_df['startTime'].dt.year == most_common_year][
                'startTime'].dt.month.value_counts().sort_index()
            print(f"Month distribution for year {most_common_year}: {month_counts.to_dict()}")

    if not predictions_df.empty and 'startTime' in predictions_df.columns and predictions_df['startTime'].notna().any():
        pred_min = predictions_df['startTime'].min()
        pred_max = predictions_df['startTime'].max()
        print(f"Predictions date range: {pred_min} to {pred_max}")

        # Add year distribution for better diagnostics
        year_counts = predictions_df['startTime'].dt.year.value_counts().sort_index()
        print(f"Predictions years distribution: {year_counts.to_dict()}")

        # Show month distribution for the most common year
        if not year_counts.empty:
            most_common_year = year_counts.idxmax()
            month_counts = predictions_df[predictions_df['startTime'].dt.year == most_common_year][
                'startTime'].dt.month.value_counts().sort_index()
            print(f"Month distribution for year {most_common_year}: {month_counts.to_dict()}")

    # Check if date ranges are compatible
    if (not ground_truth_df.empty and 'startTime' in ground_truth_df.columns and ground_truth_df[
        'startTime'].notna().any() and
            not predictions_df.empty and 'startTime' in predictions_df.columns and predictions_df[
                'startTime'].notna().any()):

        gt_years = set(ground_truth_df['startTime'].dt.year)
        pred_years = set(predictions_df['startTime'].dt.year)

        # If there's no overlap in years at all, there might be a date format issue
        if not gt_years.intersection(pred_years):
            print(f"WARNING: No overlap in years between datasets: {gt_years} vs {pred_years}")
            print("This may indicate a date format issue. Attempting to fix...")

            # Get most common years
            gt_most_common = ground_truth_df['startTime'].dt.year.mode()[0] if not ground_truth_df.empty else None
            pred_most_common = predictions_df['startTime'].dt.year.mode()[0] if not predictions_df.empty else None

            if gt_most_common and pred_most_common and abs(gt_most_common - pred_most_common) >= 10:
                print(f"Large difference in years: {gt_most_common} vs {pred_most_common}")

                # Try to determine which dataset needs fixing - usually the one with future dates
                current_year = 2025  # Approximate current year

                if gt_most_common > current_year + 1 and pred_most_common <= current_year + 1:
                    print("Ground truth dates appear to be in the future, attempting to fix month/day swap...")
                    # Fix ground truth dates
                    ground_truth_df = fix_date_format(ground_truth_df, 'startTime')
                    if 'endTime' in ground_truth_df.columns:
                        ground_truth_df = fix_date_format(ground_truth_df, 'endTime')

                elif pred_most_common > current_year + 1 and gt_most_common <= current_year + 1:
                    print("Prediction dates appear to be in the future, attempting to fix month/day swap...")
                    # Fix prediction dates
                    predictions_df = fix_date_format(predictions_df, 'startTime')
                    if 'endTime' in predictions_df.columns:
                        predictions_df = fix_date_format(predictions_df, 'endTime')

    # Add a forgiving match setting to match entries even with less time overlap
    min_overlap_pct = 0.1  # 10% overlap is enough to consider a match

    # Group by email
    for email, gt_group in ground_truth_df.groupby('email'):
        pred_group = predictions_df[predictions_df['email'] == email]

        if pred_group.empty:
            continue

        # Print date ranges for this user
        if not gt_group.empty and 'startTime' in gt_group.columns and gt_group['startTime'].notna().any():
            gt_user_min = gt_group['startTime'].min()
            gt_user_max = gt_group['startTime'].max()
            print(f"User {email} ground truth date range: {gt_user_min} to {gt_user_max}")

        if not pred_group.empty and 'startTime' in pred_group.columns and pred_group['startTime'].notna().any():
            pred_user_min = pred_group['startTime'].min()
            pred_user_max = pred_group['startTime'].max()
            print(f"User {email} predictions date range: {pred_user_min} to {pred_user_max}")

        # For each ground truth entry
        for _, gt_row in gt_group.iterrows():
            gt_start = gt_row['startTime']
            gt_end = gt_row['endTime']

            if pd.isna(gt_start):
                continue

            # If ground truth end time is missing, create a synthetic one
            if pd.isna(gt_end):
                # Try to use the next entry's start time
                next_entries = gt_group[gt_group['startTime'] > gt_start]
                if not next_entries.empty:
                    gt_end = next_entries['startTime'].min()
                else:
                    # If no next entry, set end to 24 hours after start
                    gt_end = gt_start + pd.Timedelta(hours=24)

            # Find overlapping predictions with relaxed criteria
            overlapping_preds = pred_group[
                # Prediction starts during ground truth
                ((pred_group['startTime'] >= gt_start - pd.Timedelta(minutes=30)) &
                 (pred_group['startTime'] <= gt_end + pd.Timedelta(minutes=30))) |

                # Prediction ends during ground truth
                ((pred_group['endTime'] >= gt_start - pd.Timedelta(minutes=30)) &
                 (pred_group['endTime'] <= gt_end + pd.Timedelta(minutes=30)) &
                 pred_group['endTime'].notna()) |

                # Ground truth is fully contained within prediction
                ((pred_group['startTime'] <= gt_start + pd.Timedelta(minutes=30)) &
                 (pred_group['endTime'] >= gt_end - pd.Timedelta(minutes=30)) &
                 pred_group['endTime'].notna()) |

                # Only start time matches (for predictions with NULL end time)
                ((pred_group['startTime'] >= gt_start - pd.Timedelta(minutes=30)) &
                 (pred_group['startTime'] <= gt_end + pd.Timedelta(minutes=30)) &
                 pred_group['endTime'].isna())
                ]

            for _, pred_row in overlapping_preds.iterrows():
                # Calculate overlap with tolerance
                if pd.isna(pred_row['startTime']):
                    continue

                # For predictions with NULL end time, consider them matching if start time is close
                if pd.isna(pred_row['endTime']):
                    # Check if prediction start is close to ground truth start or end
                    start_diff = abs((pred_row['startTime'] - gt_start).total_seconds())
                    end_diff = abs((pred_row['startTime'] - gt_end).total_seconds())
                    if min(start_diff, end_diff) <= 1800:  # Within 30 minutes
                        overlap_percentage = 0.5  # Assign a moderate overlap percentage
                    else:
                        continue
                else:
                    # Calculate actual overlap with tolerance
                    overlap_start = max(gt_start, pred_row['startTime'] - pd.Timedelta(minutes=15))
                    overlap_end = min(gt_end, pred_row['endTime'] + pd.Timedelta(minutes=15))

                    if overlap_end <= overlap_start:
                        continue

                    overlap_duration = (overlap_end - overlap_start).total_seconds()
                    gt_duration = (gt_end - gt_start).total_seconds()

                    if gt_duration <= 0:
                        continue

                    overlap_percentage = overlap_duration / gt_duration

                # Only consider matches with sufficient overlap
                if overlap_percentage >= min_overlap_pct:
                    match_data = {
                        'ground_truth_id': gt_row.name,
                        'email': email,
                        'overlap_percentage': overlap_percentage,
                        'gt_start': gt_start,
                        'gt_end': gt_end,
                        'pred_start': pred_row['startTime'],
                        'pred_end': pred_row['endTime'],
                        'gt_segment': gt_row['segment'] if 'segment' in gt_row else None,
                        'pred_segment': pred_row['segment'] if 'segment' in pred_row else None
                    }

                    # Add purpose information
                    if 'normalized_purpose' in gt_row:
                        match_data['gt_purpose'] = gt_row['normalized_purpose']
                    elif 'purpose' in gt_row:
                        match_data['gt_purpose'] = gt_row['purpose']

                    if 'normalized_purpose' in pred_row:
                        match_data['pred_purpose'] = pred_row['normalized_purpose']
                    elif 'purpose' in pred_row:
                        match_data['pred_purpose'] = pred_row['purpose']

                    # Add prediction ID if available
                    if 'predictionId' in pred_row:
                        match_data['prediction_id'] = pred_row['predictionId']
                    else:
                        match_data['prediction_id'] = pred_row.name

                    # Add mode information if available
                    if 'normalized_mode' in gt_row:
                        match_data['gt_mode'] = gt_row['normalized_mode']
                    elif 'mode' in gt_row:
                        match_data['gt_mode'] = gt_row['mode']

                    if 'smode' in pred_row:
                        match_data['pred_mode'] = pred_row['smode']
                    elif 'normalized_mode' in pred_row:
                        match_data['pred_mode'] = pred_row['normalized_mode']
                    elif 'mode' in pred_row:
                        match_data['pred_mode'] = extract_mode_from_json(pred_row['mode'])

                    matches.append(match_data)

    if matches:
        matches_df = pd.DataFrame(matches)
        print(f"Found {len(matches_df)} matching entries")
        return matches_df
    else:
        print("No matching entries found")
        return pd.DataFrame()


def fix_date_format(df, date_col):
    """
    Attempt to fix date format issues by swapping month and day
    """
    if date_col not in df.columns or not pd.api.types.is_datetime64_dtype(df[date_col]):
        return df

    # Create a copy to work with
    fixed_df = df.copy()

    # Current year as reference
    current_year = 2025

    # Find problematic dates (far in future)
    future_mask = fixed_df[date_col].dt.year > current_year + 1
    future_dates = fixed_df.loc[future_mask, date_col]

    if future_dates.empty:
        return fixed_df

    print(f"Attempting to fix {len(future_dates)} dates in {date_col}")

    # Try to swap month/day for each problematic date
    for idx in future_dates.index:
        original_date = fixed_df.loc[idx, date_col]
        date_str = original_date.strftime('%Y-%m-%d %H:%M:%S')
        parts = date_str.split('-')

        if len(parts) >= 3:
            year = parts[0]
            month = parts[1]
            day_time = parts[2].split(' ')
            day = day_time[0]
            time = day_time[1] if len(day_time) > 1 else "00:00:00"

            # Only swap if both month and day are valid
            if month.isdigit() and day.isdigit():
                month_int = int(month)
                day_int = int(day)

                if 1 <= month_int <= 12 and 1 <= day_int <= 12:
                    # Swap month and day
                    swapped_str = f"{year}-{day}-{month} {time}"
                    try:
                        swapped_date = pd.to_datetime(swapped_str)
                        # Only use the swapped date if it's in a plausible year
                        if swapped_date.year <= current_year + 1:
                            fixed_df.loc[idx, date_col] = swapped_date
                            print(f"Swapped date format: {date_str} -> {swapped_date}")
                    except:
                        pass

    return fixed_df


def analyze_prediction_errors(matches_df):
    """
    Analyze patterns in prediction errors, particularly for trip start detection
    """
    error_analysis = {}

    if matches_df.empty:
        return error_analysis

    # 1. Analyze trip start delays (when app detects moving later than ground truth)
    moving_matches = matches_df[matches_df['gt_segment'] == 'moving'].copy()

    if not moving_matches.empty and 'pred_segment' in moving_matches.columns:
        # Filter for entries where prediction correctly identified segment as moving
        correct_moving = moving_matches[moving_matches['pred_segment'] == 'moving']

        if not correct_moving.empty:
            # Calculate time difference between ground truth start and prediction start
            correct_moving['start_delay_seconds'] = (
                        correct_moving['pred_start'] - correct_moving['gt_start']).dt.total_seconds()

            # Calculate statistics on delays
            start_delays = correct_moving['start_delay_seconds'].dropna()

            if not start_delays.empty:
                error_analysis['trip_start_delay'] = {
                    'mean_delay_seconds': start_delays.mean(),
                    'median_delay_seconds': start_delays.median(),
                    'min_delay_seconds': start_delays.min(),
                    'max_delay_seconds': start_delays.max(),
                    'positive_delay_count': sum(start_delays > 0),  # App detected later than ground truth
                    'negative_delay_count': sum(start_delays < 0),  # App detected earlier than ground truth
                    'delay_distribution': {
                        'under_1min': sum((start_delays > 0) & (start_delays <= 60)),
                        '1-5min': sum((start_delays > 60) & (start_delays <= 300)),
                        '5-15min': sum((start_delays > 300) & (start_delays <= 900)),
                        'over_15min': sum(start_delays > 900)
                    }
                }

    # 2. Analyze segment misclassifications
    if 'gt_segment' in matches_df.columns and 'pred_segment' in matches_df.columns:
        segment_confusion = pd.crosstab(
            matches_df['gt_segment'],
            matches_df['pred_segment'],
            rownames=['Ground Truth'],
            colnames=['Prediction']
        )

        # Convert to dictionary for easy access in reports
        error_analysis['segment_confusion'] = segment_confusion.to_dict()

    # 3. Analyze mode misclassifications for moving segments
    if not moving_matches.empty and 'gt_mode' in moving_matches.columns and 'pred_mode' in moving_matches.columns:
        valid_modes = moving_matches.dropna(subset=['gt_mode', 'pred_mode'])

        if not valid_modes.empty:
            mode_confusion = pd.crosstab(
                valid_modes['gt_mode'],
                valid_modes['pred_mode'],
                rownames=['Ground Truth'],
                colnames=['Prediction']
            )

            # Convert to dictionary for easy access in reports
            error_analysis['mode_confusion'] = mode_confusion.to_dict()

    # 4. Analyze false positives/negatives for trip detection
    # Define segments before and after each moving segment in ground truth
    all_segments = matches_df[
        ['email', 'gt_segment', 'pred_segment', 'gt_start', 'gt_end', 'pred_start', 'pred_end']].dropna()

    if not all_segments.empty:
        all_segments = all_segments.sort_values(['email', 'gt_start'])

        # Calculate transitions (false negatives and false positives)
        transition_errors = {
            'missed_trips': 0,  # ground truth = moving, prediction = stationary
            'false_trips': 0,  # ground truth = stationary, prediction = moving
            'delayed_trips': 0  # correctly identified as moving but with significant delay
        }

        # Count segment mismatches
        missed_trips = all_segments[(all_segments['gt_segment'] == 'moving') &
                                    (all_segments['pred_segment'] == 'stationary')]
        transition_errors['missed_trips'] = len(missed_trips)

        false_trips = all_segments[(all_segments['gt_segment'] == 'stationary') &
                                   (all_segments['pred_segment'] == 'moving')]
        transition_errors['false_trips'] = len(false_trips)

        # Count delayed trip detections (delays over 5 minutes)
        if 'start_delay_seconds' in correct_moving.columns:
            transition_errors['delayed_trips'] = sum(correct_moving['start_delay_seconds'] > 300)

        error_analysis['transition_errors'] = transition_errors

        # Calculate percentage of each type of error
        total_segments = len(all_segments)
        if total_segments > 0:
            error_analysis['error_percentages'] = {
                'missed_trip_pct': transition_errors['missed_trips'] / total_segments * 100,
                'false_trip_pct': transition_errors['false_trips'] / total_segments * 100,
                'delayed_trip_pct': transition_errors['delayed_trips'] / total_segments * 100
            }

    return error_analysis


def calculate_metrics(matches_df):
    """
    Calculate validation metrics
    """
    metrics = {}

    # Log the number of matches found
    print(f"Number of matched entries: {len(matches_df)}")

    if matches_df.empty:
        return {
            'segment_accuracy': 0,
            'mode_accuracy': 0,
            'purpose_accuracy': 0,
            'home_work_purpose_accuracy': 0,
            'time_coverage': 0,
            'per_user': {}
        }

    # Segment accuracy
    valid_segments = matches_df.dropna(subset=['gt_segment', 'pred_segment'])
    segment_correct = valid_segments['gt_segment'] == valid_segments['pred_segment']
    metrics['segment_accuracy'] = segment_correct.mean() if len(segment_correct) > 0 else 0

    # Mode accuracy (only for moving segments)
    moving_matches = matches_df[matches_df['gt_segment'] == 'moving']
    if len(moving_matches) > 0 and 'gt_mode' in moving_matches.columns and 'pred_mode' in moving_matches.columns:
        # Filter out NaNs
        valid_modes = moving_matches.dropna(subset=['gt_mode', 'pred_mode'])
        mode_correct = valid_modes['gt_mode'] == valid_modes['pred_mode']
        metrics['mode_accuracy'] = mode_correct.mean() if len(mode_correct) > 0 else 0
    else:
        metrics['mode_accuracy'] = 0
        # Add this code in calculate_metrics right after the mode accuracy calculation
        if len(moving_matches) > 0 and 'gt_mode' in moving_matches.columns and 'pred_mode' in moving_matches.columns:
            # Print mode comparison details
            print("\n=== MODE COMPARISON DEBUGGING ===")
            print(f"Moving entries: {len(moving_matches)}")
            valid_modes = moving_matches.dropna(subset=['gt_mode', 'pred_mode'])
            print(f"Valid mode entries: {len(valid_modes)}")

            # Print sample of ground truth and prediction modes
            if not valid_modes.empty:
                print("\nSample mode comparisons (first 10):")
                for i, row in valid_modes.head(10).iterrows():
                    print(
                        f"GT: '{row['gt_mode']}' vs Pred: '{row['pred_mode']}' → Match: {row['gt_mode'] == row['pred_mode']}")

                # Count matches by mode type
                mode_matches = valid_modes[valid_modes['gt_mode'] == valid_modes['pred_mode']]
                print(
                    f"\nTotal mode matches: {len(mode_matches)}/{len(valid_modes)} ({len(mode_matches) / len(valid_modes):.2%})")

                # Create a confusion matrix for modes
                if len(valid_modes) > 0:
                    print("\nMode confusion:")
                    mode_counts = pd.crosstab(valid_modes['gt_mode'], valid_modes['pred_mode'])
                    print(mode_counts)

    # Overall purpose accuracy
    if 'gt_purpose' in matches_df.columns and 'pred_purpose' in matches_df.columns:
        valid_purposes = matches_df.dropna(subset=['gt_purpose', 'pred_purpose'])
        purpose_correct = valid_purposes['gt_purpose'] == valid_purposes['pred_purpose']
        metrics['purpose_accuracy'] = purpose_correct.mean() if len(purpose_correct) > 0 else 0
    else:
        metrics['purpose_accuracy'] = 0

    # Home and Work purpose accuracy
    if 'gt_purpose' in matches_df.columns and 'pred_purpose' in matches_df.columns:
        # Filter for entries with home or work
        home_work_df = matches_df[
            matches_df['gt_purpose'].str.lower().str.contains('home|work', na=False)
        ]

        if not home_work_df.empty:
            # Make sure to do case-insensitive comparison
            valid_hw = home_work_df.dropna(subset=['gt_purpose', 'pred_purpose'])

            # Convert both to lowercase for comparison
            valid_hw = valid_hw.copy()
            valid_hw['gt_purpose_lower'] = valid_hw['gt_purpose'].str.lower().str.strip()
            valid_hw['pred_purpose_lower'] = valid_hw['pred_purpose'].str.lower().str.strip()

            hw_matches = valid_hw['gt_purpose_lower'] == valid_hw['pred_purpose_lower']
            metrics['home_work_purpose_accuracy'] = hw_matches.mean() if len(hw_matches) > 0 else 0

            # Count matches by purpose type
            # Count matches by purpose type
            hw_counts = {
                'home_count': len(home_work_df[home_work_df['gt_purpose'].str.lower().str.contains('home', na=False)]),
                'work_count': len(home_work_df[home_work_df['gt_purpose'].str.lower().str.contains('work', na=False)]),
            }

            # For home correct matches - use case-insensitive matching
            home_entries = home_work_df[home_work_df['gt_purpose'].str.lower().str.contains('home', na=False)]
            if not home_entries.empty:
                home_entries = home_entries.copy()
                home_entries['gt_lower'] = home_entries['gt_purpose'].str.lower().str.strip()
                home_entries['pred_lower'] = home_entries['pred_purpose'].str.lower().str.strip()
                hw_counts['home_correct'] = sum(home_entries['gt_lower'] == home_entries['pred_lower'])
            else:
                hw_counts['home_correct'] = 0

            # For work correct matches - use case-insensitive matching
            work_entries = home_work_df[home_work_df['gt_purpose'].str.lower().str.contains('work', na=False)]
            if not work_entries.empty:
                work_entries = work_entries.copy()
                work_entries['gt_lower'] = work_entries['gt_purpose'].str.lower().str.strip()
                work_entries['pred_lower'] = work_entries['pred_purpose'].str.lower().str.strip()
                hw_counts['work_correct'] = sum(work_entries['gt_lower'] == work_entries['pred_lower'])
            else:
                hw_counts['work_correct'] = 0

            metrics.update(hw_counts)

    # Overall time coverage
    total_gt_duration = sum((row['gt_end'] - row['gt_start']).total_seconds()
                            for _, row in matches_df.iterrows()
                            if not pd.isna(row['gt_end']) and not pd.isna(row['gt_start']))

    total_overlap_duration = sum(
        (min(row['gt_end'], row['pred_end'] if not pd.isna(row['pred_end']) else row['gt_end']) -
         max(row['gt_start'], row['pred_start'])).total_seconds()
        for _, row in matches_df.iterrows()
        if not pd.isna(row['gt_start']) and not pd.isna(row['pred_start']))

    metrics['time_coverage'] = total_overlap_duration / total_gt_duration if total_gt_duration > 0 else 0

    # Analyze prediction errors and patterns
    error_analysis = analyze_prediction_errors(matches_df)
    metrics['error_analysis'] = error_analysis

    # Per-user metrics
    user_metrics = {}
    for email, group in matches_df.groupby('email'):
        # Segment accuracy
        valid_segments = group.dropna(subset=['gt_segment', 'pred_segment'])
        segment_accuracy = (valid_segments['gt_segment'] == valid_segments['pred_segment']).mean() if len(
            valid_segments) > 0 else 0

        # Mode accuracy for moving segments
        moving_group = group[group['gt_segment'] == 'moving']
        if len(moving_group) > 0 and 'gt_mode' in moving_group.columns and 'pred_mode' in moving_group.columns:
            valid_modes = moving_group.dropna(subset=['gt_mode', 'pred_mode'])
            mode_accuracy = (valid_modes['gt_mode'] == valid_modes['pred_mode']).mean() if len(valid_modes) > 0 else 0
        else:
            mode_accuracy = 0

        # Purpose accuracy
        if 'gt_purpose' in group.columns and 'pred_purpose' in group.columns:
            valid_purposes = group.dropna(subset=['gt_purpose', 'pred_purpose'])
            purpose_accuracy = (valid_purposes['gt_purpose'] == valid_purposes['pred_purpose']).mean() if len(
                valid_purposes) > 0 else 0
        else:
            purpose_accuracy = 0

        # Home/Work purpose accuracy
        if 'gt_purpose' in group.columns and 'pred_purpose' in group.columns:
            # Filter for entries where ground truth purpose is home or work
            home_work_group = group[
                group['gt_purpose'].str.lower().str.contains('home|work', na=False)
            ]

            if not home_work_group.empty:
                valid_hw_purposes = home_work_group.dropna(subset=['gt_purpose', 'pred_purpose'])
                hw_purpose_accuracy = (
                            valid_hw_purposes['gt_purpose'] == valid_hw_purposes['pred_purpose']).mean() if len(
                    valid_hw_purposes) > 0 else 0
            else:
                hw_purpose_accuracy = 0
        else:
            hw_purpose_accuracy = 0

        # Time coverage
        total_gt_duration = sum((row['gt_end'] - row['gt_start']).total_seconds()
                                for _, row in group.iterrows()
                                if not pd.isna(row['gt_end']) and not pd.isna(row['gt_start']))

        total_overlap_duration = sum(
            (min(row['gt_end'], row['pred_end'] if not pd.isna(row['pred_end']) else row['gt_end']) -
             max(row['gt_start'], row['pred_start'])).total_seconds()
            for _, row in group.iterrows()
            if not pd.isna(row['gt_start']) and not pd.isna(row['pred_start']))

        time_coverage = total_overlap_duration / total_gt_duration if total_gt_duration > 0 else 0

        user_metrics[email] = {
            'segment_accuracy': segment_accuracy,
            'mode_accuracy': mode_accuracy,
            'purpose_accuracy': purpose_accuracy,
            'home_work_purpose_accuracy': hw_purpose_accuracy,
            'time_coverage': time_coverage,
            'num_entries': len(group)
        }

    metrics['per_user'] = user_metrics

    return metrics


def generate_report(metrics, matches_df, output_dir):
    """
    Generate validation report and save it to CSV
    """
    os.makedirs(output_dir, exist_ok=True)

    # Overall metrics report
    overall_report = {
        'Metric': ['Segment Accuracy', 'Mode Accuracy', 'Purpose Accuracy',
                   'Home/Work Purpose Accuracy', 'Time Coverage', 'Matched Entries'],
        'Value': [
            metrics['segment_accuracy'],
            metrics['mode_accuracy'],
            metrics.get('purpose_accuracy', 0),
            metrics.get('home_work_purpose_accuracy', 0),
            metrics['time_coverage'],
            len(matches_df)
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
    overall_df.to_csv(os.path.join(output_dir, 'overall_metrics.csv'), index=False)

    # Per-user metrics report
    user_rows = []
    for email, user_metric in metrics['per_user'].items():
        user_rows.append({
            'Email': email,
            'Segment_Accuracy': user_metric['segment_accuracy'],
            'Mode_Accuracy': user_metric['mode_accuracy'],
            'Purpose_Accuracy': user_metric.get('purpose_accuracy', 0),
            'Home_Work_Purpose_Accuracy': user_metric.get('home_work_purpose_accuracy', 0),
            'Time_Coverage': user_metric['time_coverage'],
            'Num_Entries': user_metric.get('num_entries', 0)  # Default to 0 if not present
        })

    user_df = pd.DataFrame(user_rows)
    if not user_df.empty:
        user_df.to_csv(os.path.join(output_dir, 'user_metrics.csv'), index=False)

    # Error analysis report
    if 'error_analysis' in metrics:
        error_analysis = metrics['error_analysis']

        # Trip start delay analysis
        if 'trip_start_delay' in error_analysis:
            delay_data = error_analysis['trip_start_delay']
            delay_df = pd.DataFrame({
                'Metric': ['Mean Delay (sec)', 'Median Delay (sec)', 'Min Delay (sec)', 'Max Delay (sec)',
                           'Positive Delay Count', 'Negative Delay Count',
                           'Delay < 1min', 'Delay 1-5min', 'Delay 5-15min', 'Delay > 15min'],
                'Value': [
                    delay_data['mean_delay_seconds'],
                    delay_data['median_delay_seconds'],
                    delay_data['min_delay_seconds'],
                    delay_data['max_delay_seconds'],
                    delay_data['positive_delay_count'],
                    delay_data['negative_delay_count'],
                    delay_data['delay_distribution']['under_1min'],
                    delay_data['delay_distribution']['1-5min'],
                    delay_data['delay_distribution']['5-15min'],
                    delay_data['delay_distribution']['over_15min']
                ]
            })
            delay_df.to_csv(os.path.join(output_dir, 'trip_start_delay_analysis.csv'), index=False)

        # Transition errors
        if 'transition_errors' in error_analysis:
            errors = error_analysis['transition_errors']
            percentages = error_analysis.get('error_percentages', {})

            error_df = pd.DataFrame({
                'Error Type': ['Missed Trips', 'False Trips', 'Delayed Trips'],
                'Count': [
                    errors['missed_trips'],
                    errors['false_trips'],
                    errors['delayed_trips']
                ],
                'Percentage': [
                    percentages.get('missed_trip_pct', 0),
                    percentages.get('false_trip_pct', 0),
                    percentages.get('delayed_trip_pct', 0)
                ]
            })
            error_df.to_csv(os.path.join(output_dir, 'transition_errors.csv'), index=False)

        # Segment confusion matrix
        if 'segment_confusion' in error_analysis:
            segment_confusion = pd.DataFrame(error_analysis['segment_confusion'])
            segment_confusion.to_csv(os.path.join(output_dir, 'segment_confusion.csv'))

        # Mode confusion matrix
        if 'mode_confusion' in error_analysis:
            mode_confusion = pd.DataFrame(error_analysis['mode_confusion'])
            mode_confusion.to_csv(os.path.join(output_dir, 'mode_confusion.csv'))

    # Detailed matches report
    if not matches_df.empty:
        # Save the full detailed matches DataFrame
        matches_df.to_csv(os.path.join(output_dir, 'detailed_matches.csv'), index=False)

        # Create a more simplified view for easier analysis
        simple_matches = matches_df[['email', 'gt_start', 'gt_end', 'pred_start', 'pred_end',
                                     'gt_segment', 'pred_segment', 'overlap_percentage']]

        if 'gt_mode' in matches_df.columns and 'pred_mode' in matches_df.columns:
            simple_matches['gt_mode'] = matches_df['gt_mode']
            simple_matches['pred_mode'] = matches_df['pred_mode']
            simple_matches['mode_match'] = matches_df['gt_mode'] == matches_df['pred_mode']

        if 'gt_purpose' in matches_df.columns and 'pred_purpose' in matches_df.columns:
            simple_matches['gt_purpose'] = matches_df['gt_purpose']
            simple_matches['pred_purpose'] = matches_df['pred_purpose']
            simple_matches['purpose_match'] = matches_df['gt_purpose'] == matches_df['pred_purpose']

            # Add home/work purpose specific analysis
            is_home_work = simple_matches['gt_purpose'].str.lower().str.contains('home|work', na=False)
            simple_matches['is_home_work'] = is_home_work
            simple_matches['home_work_match'] = False
            simple_matches.loc[is_home_work, 'home_work_match'] = simple_matches.loc[is_home_work, 'purpose_match']

        simple_matches['segment_match'] = matches_df['gt_segment'] == matches_df['pred_segment']

        # Add start time difference for trip delay analysis
        if 'gt_start' in simple_matches.columns and 'pred_start' in simple_matches.columns:
            simple_matches['start_time_diff_seconds'] = (
                        simple_matches['pred_start'] - simple_matches['gt_start']).dt.total_seconds()

        simple_matches.to_csv(os.path.join(output_dir, 'simple_matches.csv'), index=False)

    print(f"Reports saved to {output_dir}")

    # Print a summary
    print("\n===== VALIDATION SUMMARY =====")
    print(f"Segment Accuracy: {metrics['segment_accuracy']:.2%}")
    print(f"Mode Accuracy: {metrics['mode_accuracy']:.2%}")
    if 'purpose_accuracy' in metrics:
        print(f"Purpose Accuracy: {metrics['purpose_accuracy']:.2%}")
    if 'home_work_purpose_accuracy' in metrics:
        print(f"Home/Work Purpose Accuracy: {metrics['home_work_purpose_accuracy']:.2%}")
    print(f"Time Coverage: {metrics['time_coverage']:.2%}")
    print(f"Total Matched Entries: {len(matches_df)}")
    print(f"Number of Users: {len(metrics['per_user'])}")

    # Print error analysis summary
    if 'error_analysis' in metrics and 'trip_start_delay' in metrics['error_analysis']:
        delay_data = metrics['error_analysis']['trip_start_delay']
        print("\n--- Trip Start Delay Analysis ---")
        print(f"Mean Delay: {delay_data['mean_delay_seconds']:.1f} seconds")
        print(f"Median Delay: {delay_data['median_delay_seconds']:.1f} seconds")
        print(
            f"Trips with Delay > 5 min: {delay_data['delay_distribution']['5-15min'] + delay_data['delay_distribution']['over_15min']}")

    if 'error_analysis' in metrics and 'transition_errors' in metrics['error_analysis']:
        errors = metrics['error_analysis']['transition_errors']
        print("\n--- Trip Detection Errors ---")
        print(f"Missed Trips: {errors['missed_trips']}")
        print(f"False Trips: {errors['false_trips']}")
        print(f"Delayed Trips (>5min): {errors['delayed_trips']}")

    print("=============================")


def validate_file_pair(ground_truth_path, predictions_path, output_dir):
    """
    Validate a pair of files
    """
    basename = os.path.splitext(os.path.basename(ground_truth_path))[0]
    user_output_dir = os.path.join(output_dir, basename)
    os.makedirs(user_output_dir, exist_ok=True)

    print(f"\nProcessing file pair: {basename}")
    result = validate_single_files(ground_truth_path, predictions_path, user_output_dir)

    if result:
        return result[1]  # Return the matches DataFrame
    return None


def validate_single_files(ground_truth_path, predictions_path, output_dir):
    """
    Validate single ground truth and predictions files
    """
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

    print(f"Ground truth entries: {len(ground_truth_df)}")
    print(f"Prediction entries: {len(predictions_df)}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Normalize data
    print("Normalizing data...")
    # Normalize dates (adjust timezone for predictions)
    ground_truth_df = normalize_dates(ground_truth_df, ['startTime', 'endTime'], adjust_timezone=False)
    predictions_df = normalize_dates(predictions_df, ['startTime', 'endTime'], adjust_timezone=True)

    # Normalize segments
    ground_truth_df = normalize_segment_values(ground_truth_df, 'segment')
    predictions_df = normalize_segment_values(predictions_df, 'segment')

    # Normalize purposes
    if 'purpose' in ground_truth_df.columns:
        ground_truth_df = normalize_purpose_values(ground_truth_df, 'purpose')
    if 'purpose' in predictions_df.columns:
        predictions_df = normalize_purpose_values(predictions_df, 'purpose')

    # Normalize modes
    ground_truth_df, predictions_df = normalize_mode_values(ground_truth_df, predictions_df)

    # Normalize durations
    ground_truth_df, predictions_df = normalize_durations(ground_truth_df, predictions_df)

    # Find matches
    print("Finding overlapping entries...")
    matches_df = find_overlapping_entries(ground_truth_df, predictions_df)

    if matches_df.empty:
        print("No matching entries found")
        return None

    # Calculate metrics
    print("Calculating validation metrics...")
    metrics = calculate_metrics(matches_df)

    # Generate report
    print("Generating validation report...")
    generate_report(metrics, matches_df, output_dir)

    return metrics, matches_df


def validate_travel_diaries(ground_truth_path, predictions_path, output_dir='validation_results'):
    """
    Main function to validate travel diaries
    """
    # Check if inputs are directories or files
    if os.path.isfile(ground_truth_path) and os.path.isfile(predictions_path):
        print("Processing single files mode")
        return validate_single_files(ground_truth_path, predictions_path, output_dir)

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
            matched_files = [(gt_basenames[name], pred_basenames[name]) for name in common_basenames]

            # Process each file pair
            all_matches = []
            for gt_file, pred_file in matched_files:
                matches = validate_file_pair(gt_file, pred_file, output_dir)
                if matches is not None:
                    all_matches.append(matches)

            if all_matches:
                all_matches_df = pd.concat(all_matches, ignore_index=True)
                overall_metrics = calculate_metrics(all_matches_df)
                generate_report(overall_metrics, all_matches_df, output_dir)
                return overall_metrics, all_matches_df

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each user
    all_matches = []
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

        print(f"Ground truth entries: {len(ground_truth_df)}")
        print(f"Prediction entries: {len(predictions_df)}")

        # Normalize data
        print("Normalizing data...")
        # Normalize dates (adjust timezone for predictions)
        ground_truth_df = normalize_dates(ground_truth_df, ['startTime', 'endTime'], adjust_timezone=False)
        predictions_df = normalize_dates(predictions_df, ['startTime', 'endTime'], adjust_timezone=True)

        # Normalize segments
        ground_truth_df = normalize_segment_values(ground_truth_df, 'segment')
        predictions_df = normalize_segment_values(predictions_df, 'segment')

        # Normalize purposes
        if 'purpose' in ground_truth_df.columns:
            ground_truth_df = normalize_purpose_values(ground_truth_df, 'purpose')
        if 'purpose' in predictions_df.columns:
            predictions_df = normalize_purpose_values(predictions_df, 'purpose')

        # Normalize modes
        ground_truth_df, predictions_df = normalize_mode_values(ground_truth_df, predictions_df)

        # Normalize durations
        ground_truth_df, predictions_df = normalize_durations(ground_truth_df, predictions_df)

        # Find matches
        print("Finding overlapping entries...")
        matches_df = find_overlapping_entries(ground_truth_df, predictions_df)

        if matches_df.empty:
            print(f"No matching entries found for user {user}")
            continue

        # Add user to all matches
        all_matches.append(matches_df)

        # Calculate metrics for this user
        print(f"Calculating metrics for user {user}")
        user_metric = calculate_metrics(matches_df)
        user_metrics[user] = user_metric

        # Generate user-specific report
        user_output_dir = os.path.join(output_dir, user.split('@')[0])
        os.makedirs(user_output_dir, exist_ok=True)
        print(f"Generating report for user {user}")
        generate_report(user_metric, matches_df, user_output_dir)

    # Combine all matches and calculate overall metrics
    if all_matches:
        all_matches_df = pd.concat(all_matches, ignore_index=True)
        print("\nCalculating overall metrics...")
        overall_metrics = calculate_metrics(all_matches_df)

        # Add per-user metrics
        overall_metrics['per_user'] = user_metrics

        # Generate overall report
        print("Generating overall validation report...")
        generate_report(overall_metrics, all_matches_df, output_dir)

        return overall_metrics, all_matches_df
    else:
        print("No matching entries found across all users")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate travel diary predictions against ground truth')
    parser.add_argument('--ground-truth', required=True, help='Path to folder containing ground truth CSV files')
    parser.add_argument('--predictions', required=True, help='Path to folder containing prediction CSV files')
    parser.add_argument('--output', default='validation_results', help='Output directory for validation results')

    args = parser.parse_args()

    validate_travel_diaries(args.ground_truth, args.predictions, args.output)