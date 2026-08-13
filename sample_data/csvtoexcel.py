import pandas as pd
import os

def csv_to_excel(csv_file_path, excel_file_path):
    """
    Convert CSV file to Excel sheet using pandas.
    
    Args:
        csv_file_path (str): Path to the input CSV file
        excel_file_path (str): Path to the output Excel file
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_file_path)
        
        # Write to Excel file
        df.to_excel(excel_file_path, index=False, sheet_name='Sheet1')
        
        print(f"Successfully converted {csv_file_path} to {excel_file_path}")
    except FileNotFoundError:
        print(f"Error: File {csv_file_path} not found")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Example usage
    csv_file = "D:\Project\Ai-Data-Insight-System\sample_data\Salary_Data.csv"
    excel_file = "Salary-Data.xlsx"
    
    csv_to_excel(csv_file, excel_file)