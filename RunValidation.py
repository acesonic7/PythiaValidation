"""
Example usage of the Travel Diary Validation Script

This script demonstrates how to use the validation script with folders of sample data.
It supports both entry-based matching validation and time-slice validation methods.

Entry-based matching: Matches overlapping entries between ground truth and predictions
Time-slice validation: Evaluates prediction accuracy at fixed time intervals (5-minute slices by default)
"""
import os
import argparse
from PythiaValidation import validate_travel_diaries
# Import the time-slice validation method
from TimeSlice import timeslice_validation


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Run travel diary validation with different methods')
    parser.add_argument('--method', choices=['entry', 'timeslice'], default='entry',
                      help='Validation method: entry-based matching or time-slice evaluation')
    parser.add_argument('--ground-truth', default='GroundTruth',
                      help='Path to ground truth folder or file')
    parser.add_argument('--predictions', default='Predictions',
                      help='Path to predictions folder or file')
    parser.add_argument('--output', default='validation_results',
                      help='Output directory for validation results')
    parser.add_argument('--interval', type=int, default=5,
                      help='Time slice interval in minutes (for time-slice method only)')
    args = parser.parse_args()

    # Define paths to the input folders and output directory
    ground_truth_folder = args.ground_truth
    predictions_folder = args.predictions
    output_dir = args.output

    # Ensure the data directories exist
    os.makedirs(ground_truth_folder, exist_ok=True)
    os.makedirs(predictions_folder, exist_ok=True)

    # Run validation using the selected method
    if args.method == 'entry':
        print(f"Running entry-based validation between {ground_truth_folder} and {predictions_folder}")
        result = validate_travel_diaries(ground_truth_folder, predictions_folder, output_dir)
    else:  # timeslice method
        print(f"Running time-slice validation between {ground_truth_folder} and {predictions_folder}")
        print(f"Using {args.interval}-minute time slices")
        result = timeslice_validation(ground_truth_folder, predictions_folder,
                                     f"{output_dir}_timeslice", args.interval)

    if result:
        metrics, data = result  # data is matches_df for entry method, slices_df for timeslice method

        # You can access the metrics programmatically
        print("\nAccessing metrics programmatically:")
        print(f"Overall segment accuracy: {metrics['segment_accuracy']:.2%}")
        print(f"Overall mode accuracy: {metrics['mode_accuracy']:.2%}")

        if 'purpose_accuracy' in metrics:
            print(f"Overall purpose accuracy: {metrics['purpose_accuracy']:.2%}")

        if 'home_work_purpose_accuracy' in metrics:
            print(f"Home/Work purpose accuracy: {metrics['home_work_purpose_accuracy']:.2%}")

        # Display number of matches or time slices
        if args.method == 'entry':
            print(f"\nTotal matched entries: {len(data)}")
        else:
            print(f"\nTotal time slices evaluated: {len(data)}")

        # Example: Get the user with the highest segment accuracy
        if metrics['per_user']:
            best_user = max(metrics['per_user'].items(),
                            key=lambda x: x[1]['segment_accuracy'])
            print(f"\nUser with highest segment accuracy: {best_user[0]}")
            print(f"Accuracy: {best_user[1]['segment_accuracy']:.2%}")

            # Example: Print all users' segment accuracy
            print("\nAll users' segment accuracy:")
            for user, user_metrics in metrics['per_user'].items():
                print(f"{user}: {user_metrics['segment_accuracy']:.2%}")

        # Method-specific results
        if args.method == 'timeslice' and 'error_analysis' in metrics and 'hourly_accuracy' in metrics['error_analysis']:
            hourly = metrics['error_analysis']['hourly_accuracy']
            print("\nHourly segment accuracy:")
            for hour_data in sorted(hourly, key=lambda x: x['hour']):
                print(f"Hour {hour_data['hour']:02d}: {hour_data['segment_accuracy']:.2%}")
    else:
        print("Validation failed or no matches/time slices found.")


if __name__ == "__main__":
    main()