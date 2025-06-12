# data_cleaning.py for dataset cleaning
from difflib import SequenceMatcher
from IPython.display import display, HTML
import numpy as np
import pandas as pd
import re
from tqdm import tqdm


# Function to identify non-standard missing values in object-type columns
def check_existing_missing_values(df):
    """
    Checks object-type columns in a DataFrame for non-standard missing values.

    Parameters:
    df (DataFrame): The dataset to inspect.

    Output:
    Displays the number of non-standard missing entries per column and the matched values.
    """

    # Common non-standard representations of missing values
    missing_values = ['', ' ', 'N/A', 'none', 'None', 'null', 'NULL', 'NaN', 'nan', 'NAN', 'nat', 'NaT']

    display(HTML(f"<h4>Scanning for Non-Standard Missing Values</h4>"))

    for column in df.columns:
        if df[column].dtype != 'object':
            continue

        matches = df[df[column].isin(missing_values)][column].unique()

        if df[column].isin(missing_values).any() and matches.size > 0:
            count = df[column].isin(missing_values).sum()
            display(HTML(f"> Missing values in column <i>'{column}'</i>: <b>{count}</b>"))
            display(HTML(f"&emsp;Matched non-standard values: {list(matches)}"))
        else:
            display(HTML(f"> Missing values in column <i>'{column}'</i>: <b>None</b>"))

    print()
    
    return None

# Function to standardize non-standard missing values to pd.NA
def replace_missing_values(df, include=None, exclude=None):
    """
    Replaces common non-standard missing value entries in object-type columns with pd.NA.

    Parameters:
    df (DataFrame): The input dataset.
    include (list, optional): List of columns to include. If None, all columns except those in 'exclude' are considered.
    exclude (list, optional): List of columns to exclude from replacement.

    Returns:
    DataFrame: Updated DataFrame with non-standard missing values replaced by pd.NA.
    """

    missing_values = ['', ' ', 'N/A', 'none', 'None', 'null', 'NULL', 'NaN', 'nan', 'NAN', 'nat', 'NaT']

    if exclude is None:
        exclude = []

    if include is None:
        available_columns = [col for col in df.columns if col not in exclude]
    else:
        available_columns = [col for col in include if col not in exclude]

    for column in available_columns:
        if df[column].dtype == 'object' and df[column].isin(missing_values).any():
            df[column] = df[column].replace(missing_values, pd.NA)

    return df

# Function to normalize string formatting in object-type columns
def normalize_string_format(df, include=None, exclude=None):
    """
    Standardizes text formatting for object-type (string) columns in a DataFrame.

    Operations performed:
    - Converts text to lowercase
    - Strips leading/trailing whitespace
    - Replaces punctuation with spaces
    - Collapses spaces into underscores
    - Removes redundant underscores

    Parameters:
    df (DataFrame): The input DataFrame.
    include (list, optional): Specific columns to apply formatting to. If None, applies to all except those in 'exclude'.
    exclude (list, optional): Columns to skip.

    Returns:
    DataFrame: Updated DataFrame with normalized string formats.
    """
    
    if exclude is None:
        exclude = []

    if include is None:
        available_columns = [col for col in df.columns if col not in exclude]
    else:
        available_columns = [col for col in include if col not in exclude]

    for column in available_columns:
        if df[column].dtype == 'object':
            df[column] = df[column].str.lower()
            df[column] = df[column].str.strip()
            df[column] = df[column].str.replace(r'[^\w\s]', ' ', regex=True)
            df[column] = df[column].str.replace(r'\s+', '_', regex=True)
            df[column] = df[column].str.replace(r'__+', '_', regex=True)

    return df

# Function to normalize column header formats for consistency
def normalize_columns_headers_format(df):
    """
    Standardizes the format of column headers in a DataFrame by:
    - Converting to lowercase
    - Stripping leading/trailing whitespace
    - Replacing punctuation with spaces
    - Collapsing spaces into underscores
    - Removing redundant underscores

    Parameters:
    df (DataFrame): The input DataFrame.

    Returns:
    DataFrame: DataFrame with cleaned and standardized column names.
    """

    title_norm = {}

    for title in df.columns:
        nt = title.lower().strip()
        nt = re.sub(r'[^\w\s]', ' ', nt)
        nt = re.sub(r'\s+', '_', nt)
        nt = re.sub(r'__+', '_', nt)
        title_norm[title] = nt

    df = df.rename(columns=title_norm)

    return df

# Function to detect potential implicit duplicates using fuzzy matching and normalization
def detect_implicit_duplicates(df, include=None, exclude=None, fuzzy_threshold=0.85):
    """
    Identifies implicit (non-exact) duplicates within string-based columns using normalization,
    token splitting, and fuzzy matching.

    Parameters:
    df (DataFrame): The input dataset.
    include (list, optional): Specific columns to check. If None, all columns are considered except those in 'exclude'.
    exclude (list, optional): Columns to ignore during processing.
    fuzzy_threshold (float): Minimum similarity ratio (0 to 1) for fuzzy matching.

    Output:
    Displays lists of entries in each column that are likely to be semantically or visually duplicated.
    """

    def normalize(value):
        """Converts string to lowercase and removes non-alphanumeric characters."""
        return re.sub(r'\W+', '', value.lower()) if isinstance(value, str) else ''

    def split_words(value):
        """Splits a string into alphanumeric tokens, including camelCase or underscore-separated parts."""
        if not isinstance(value, str):
            return []
        return re.findall(r'[A-Za-z0-9]+', value.lower())

    def fuzzy_match(a, b):
        """Returns True if similarity between two strings exceeds the defined threshold."""
        return SequenceMatcher(None, a, b).ratio() >= fuzzy_threshold

    display(HTML(f"<h4>Scanning for Implicit Duplicates</h4>"))

    if include:
        columns = [col for col in include if col in df.columns]
    elif exclude:
        columns = [col for col in df.columns if col not in exclude]
    else:
        columns = df.columns.tolist()

    for col in columns:
        display(HTML(f"<br><b>Processing column:</b> <i>{col}</i>"))

        values = df[col].dropna().unique()
        values = [v for v in values if isinstance(v, str)]
        normalized_values = {v: normalize(v) for v in values}
        results = {}

        for base in tqdm(values, desc=f"Comparing column '{col}'", unit=" values"):
            base_norm = normalized_values[base]
            base_parts = set(split_words(base))
            matches = []

            for other in values:
                if base == other:
                    continue
                other_norm = normalized_values[other]
                other_parts = set(split_words(other))

                if (
                    base_norm in other_norm or 
                    other_norm in base_norm or 
                    base_parts & other_parts or 
                    fuzzy_match(base_norm, other_norm)
                ):
                    matches.append(other)

            if matches:
                results[base] = matches

        display(HTML(f"<br><b>Results for column:</b> <i>{col}</i>"))
        if results:
            for base, found in results.items():
                display(HTML(f"<b>{base}</b> → {found}"))
        else:
            display(HTML("No implicit duplicates were found."))

    return None

# Function to convert string-based date/time columns to timezone-aware datetime or time objects
def normalize_datetime(df, include=None, exclude=None, frmt=None, time_zone='UTC'):
    """
    Converts string-based columns in a DataFrame to datetime or time objects,
    with optional format and timezone adjustments.

    Parameters:
    df (DataFrame): The input DataFrame.
    include (list, optional): Specific columns to include. If None, all non-excluded columns are processed.
    exclude (list, optional): Columns to exclude from conversion.
    frmt (str, optional): Optional datetime format (e.g., '%Y-%m-%d', '%H:%M:%S').
    time_zone (str): Timezone to localize or convert to (default: 'UTC').

    Returns:
    DataFrame: DataFrame with parsed datetime or time columns.
    """

    if exclude is None:
        exclude = []

    if include is None:
        target_columns = [col for col in df.columns if col not in exclude]
    else:
        target_columns = [col for col in include if col not in exclude]

    for column in target_columns:
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            df[column] = pd.to_datetime(df[column], format=frmt, errors='coerce')

        if pd.api.types.is_datetime64_any_dtype(df[column]):
            if frmt in ["%H:%M:%S", "%H:%M"]:
                df[column] = df[column].dt.time
            else:
                if df[column].dt.tz is None:
                    df[column] = df[column].dt.tz_localize(time_zone)
                else:
                    df[column] = df[column].dt.tz_convert(time_zone)

    return df

# Function to identify values that fail numeric conversion or are not whole numbers
def find_fail_conversion_to_numeric(df, column):
    """
    Identifies problematic entries in a column during numeric conversion.

    Checks for:
    - Non-numeric string values that cannot be converted to numbers.
    - Numeric values that are not whole integers (i.e., decimals).

    Parameters:
    df (DataFrame): The input DataFrame.
    column (str): The name of the column to analyze.

    Output:
    Prints non-numeric values and numeric values that are not integers.
    """

    # Find non-numeric values (e.g., strings, symbols)
    mask = pd.to_numeric(df[column], errors='coerce').isna() & df[column].notna()
    non_numeric_values = df.loc[mask, column]

    if not non_numeric_values.empty:
        print(f"> Non-numeric values found in column '{column}':")
        print(non_numeric_values)
        print(f"> Total non-numeric entries: {non_numeric_values.shape[0]}\n")

    # Find numeric values that are not integers
    numeric_col = pd.to_numeric(df[column], errors='coerce')
    decimal_mask = (numeric_col % 1 != 0) & (~numeric_col.isna())
    non_integer_values = df.loc[decimal_mask, column]

    if not non_integer_values.empty:
        print(f"> Numeric values that are not whole integers found in column '{column}':")
        print(non_integer_values)
        print(f"> Total non-integer entries: {non_integer_values.shape[0]}\n")
    
    return None

# Function to convert columns to numeric types (integer or float) with error detection
def convert_object_to_numeric(df, type=None, include=None, exclude=None):
    """
    Converts specified DataFrame columns to numeric types, with optional control over integer vs float conversion.

    Parameters:
    df (DataFrame): The input dataset.
    type (str, optional): Specify 'integer', 'float', or None for automatic conversion.
    include (list, optional): List of columns to convert. If None, all columns are considered except those in 'exclude'.
    exclude (list, optional): Columns to exclude from conversion.

    Returns:
    DataFrame: The updated DataFrame with converted numeric columns.
    """

    if exclude is None:
        exclude = []

    if include is None:
        available_columns = [col for col in df.columns if col not in exclude]
    else:
        available_columns = [col for col in include if col not in exclude]

    for column in available_columns:
        # Integer conversion
        if type == 'integer':
            try:
                if np.array_equal(df[column], df[column].astype(int)):
                    df[column] = pd.to_numeric(df[column], downcast='integer', errors='coerce')
                else:
                    find_errors_to_numeric(df, column)
            except Exception:
                find_errors_to_numeric(df, column)

        # Float conversion
        elif type == 'float':
            df[column] = pd.to_numeric(df[column], downcast='float', errors='coerce')

        # Auto conversion
        else:
            try:
                if np.array_equal(df[column], df[column].astype(int)):
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                else:
                    find_errors_to_numeric(df, column)
                    df[column] = pd.to_numeric(df[column], errors='coerce')
            except Exception:
                find_errors_to_numeric(df, column)
                df[column] = pd.to_numeric(df[column], errors='coerce')

    return df

# Function to convert integer columns to boolean (True/False)
def convert_integer_to_boolean(df, include=None, exclude=None):
    """
    Converts eligible integer-type columns in a DataFrame to boolean (True/False),
    where non-zero values become True and zeros become False.

    Parameters:
    df (DataFrame): The input dataset.
    include (list, optional): Specific columns to convert. If None, all columns except those in 'exclude' are evaluated.
    exclude (list, optional): Columns to skip during conversion.

    Returns:
    DataFrame: DataFrame with specified columns converted to boolean where applicable.
    """

    if exclude is None:
        exclude = []

    if include is None:
        available_columns = [col for col in df.columns if col not in exclude]
    else:
        available_columns = [col for col in include if col not in exclude]

    for column in available_columns:
        if pd.api.types.is_integer_dtype(df[column]):
            df[column] = df[column].astype(bool)

    return df

# Function to convert abbreviated gender values (e.g., 'm', 'f') to full terms ('male', 'female')
def standardize_gender_values(df, include=None, exclude=None):
    """
    Standardizes gender representations in object-type columns by converting
    abbreviations like 'm' and 'f' to 'male' and 'female'.

    Parameters:
    df (DataFrame): The input dataset.
    include (list, optional): Specific columns to apply conversion. If None, all non-excluded object-type columns are used.
    exclude (list, optional): Columns to skip during conversion.

    Returns:
    DataFrame: DataFrame with gender values standardized to full descriptors.
    """

    if exclude is None:
        exclude = []

    if include is None:
        available_columns = [col for col in df.columns if col not in exclude]
    else:
        available_columns = [col for col in include if col not in exclude]

    for column in available_columns:
        if df[column].dtype == 'object':
            df[column] = df[column].replace({'f': 'female', 'm': 'male'})

    return df



    