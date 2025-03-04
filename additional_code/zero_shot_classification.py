import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
import math
import pandas as pd
import plotly.express as px
from datetime import datetime
import openai
from tqdm.auto import tqdm
import json
from additional_code.data_clean import read_and_clean_data

def classify_conflict_related(df, text_column="notes", batch_size=16):
    """
    Classify each row in 'df' against multiple candidate labels, returning
    independent confidence scores (no softmax). For each label, the model
    produces a score in [0, 1] indicating how likely it is that the text
    matches that label.

    This function creates:
      - A column for each candidate label, storing the confidence score.
      - A 'predicted_label' column picking whichever label has the highest score
        (optional, for convenience).

    Args:
        df (pd.DataFrame): DataFrame with a column containing text for classification.
        text_column (str): Column name that holds the text to classify.
        batch_size (int): How many rows to process at once for efficiency.

    Returns:
        pd.DataFrame: The original DataFrame with new columns for label scores
                      plus a 'predicted_label' column.
    """

    # Initialize the zero-shot classification pipeline
    clf_pipeline = pipeline(
        "zero-shot-classification",
        model="roberta-large-mnli",  # or another zero-shot-capable NLI model
        device=0 if torch.cuda.is_available() else -1
    )

    # Define your candidate labels
    # (Add as many as you need, each label is checked independently)
    candidate_labels = [
        "This text describes a protest related to the Israel-Palestine conflict",
        "This text describes a protest unrelated to the Israel-Palestine conflict",
        "This text describes a pro-Israel protest",
        "This text describes a pro-Palestine protest"
    ]

    # A helper function to classify one batch of texts
    def classify_batch(texts):
        # multi_label=True => The pipeline uses a sigmoid for each label
        # The scores will NOT sum to 1
        outputs = clf_pipeline(
            texts,
            candidate_labels=candidate_labels,
            multi_label=True
        )

        batch_scores = []
        for out in outputs:
            # out["labels"] is sorted by descending score
            # out["scores"] is sorted in the same order
            label2score = {lab: sc for lab, sc in zip(out["labels"], out["scores"])}
            # Retrieve scores in the *original candidate_labels order*
            # so each row is [score_for_label_0, score_for_label_1, etc.]
            scores_in_order = [label2score.get(lbl, 0.0) for lbl in candidate_labels]
            batch_scores.append(scores_in_order)
        
        return batch_scores
    
    

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the DataFrame.")

    all_scores = []
    total_batches = math.ceil(len(df) / batch_size)
    # Process in batches
    for start_idx in tqdm(range(0, len(df), batch_size),
                          desc="Classifying",
                          total=total_batches):
        end_idx = start_idx + batch_size
        batch_texts = df[text_column][start_idx:end_idx].tolist()
        batch_scores = classify_batch(batch_texts)
        all_scores.extend(batch_scores)


    row_names = ['conflict_related', 'not_conflict_related', 'pro_israel', 'pro_palestine']
    # For each candidate label, create a new column with scores
    for i, label in enumerate(row_names):
        # Rename columns in a way that’s easy to interpret
        col_name = label
        df[col_name] = [row[i] for row in all_scores]

    # # (Optional) Choose which label had the highest confidence
    # # so you have a single predicted_label column
    # def pick_label(scores_list):
    #     max_index = max(range(len(scores_list)), key=lambda idx: scores_list[idx])
    #     return candidate_labels[max_index]

    # df["predicted_label"] = [pick_label(row) for row in all_scores]

    return df


import toml
import os
import openai
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil
import json

# Import the rate-limiting functionality
from ratelimit import limits, sleep_and_retry
import time
import random
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

def classify_protest_openai_parallel(
    df, 
    text_column="notes", 
    openai_api_key=None, 
    model="gpt-4o-mini",
    backup_file=None,
    backup_every_n=1000,
    max_workers=5,         # Reduced from 20 to 10 to avoid rate limits
    chunk_size=500,         # How many rows to process in each chunk before a backup flush
    batch_size=15,          # Increased from 10 to 15 for more efficient batching
    rate_limit_tpm=180000   # Tokens per minute rate limit (adjust to your plan)
):
    """
    Classify each row in 'df' in parallel using the OpenAI API, with batching.

    For each row, the classification label should be:
      - 'pro-Israel'
      - 'pro-Palestinian'
      - 'unrelated'
    If there's Jew-hatred, append ' antisemite' to that label.

    Creates columns:
        - label
        - conflict_related        (1 if 'pro-Israel' or 'pro-Palestinian', else 0)
        - not_conflict_related    (1 if 'unrelated', else 0)
        - pro_israel              (1 if 'pro-Israel', else 0)
        - pro_palestine           (1 if 'pro-Palestinian', else 0)
        - antisemite              (1 if 'antisemite', else 0)

    This version does the classification in parallel with batching and proper rate limit handling.
    We chunk the dataframe, create batches within each chunk, classify each batch in parallel with
    exponential backoff for rate limits, then optionally flush partial progress to a backup after 
    each chunk.

    Args:
        df (pd.DataFrame): DataFrame containing text data.
        text_column (str): Column name in 'df' with the text to classify.
        openai_api_key (str): Your OpenAI API key.
        model (str): Model name (e.g. 'gpt-4o-mini').
        backup_file (str): Path to a CSV file for saving/restoring partial progress.
        backup_every_n (int): How many new classifications to accumulate before writing to disk.
        max_workers (int): Number of parallel workers to run. Adjust to avoid rate-limit issues.
        chunk_size (int): Number of rows to process in each chunk before a backup flush.
        batch_size (int): Number of samples to classify in a single API request.
        rate_limit_tpm (int): Tokens per minute rate limit for your OpenAI account.

    Returns:
        pd.DataFrame: DataFrame with classification columns added/updated.
    """
    # --------------------
    # 1. Basic checks
    # --------------------
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the DataFrame.")
    if openai_api_key:
        openai.api_key = openai_api_key

    # -----------------------------
    # 2. Prepare classification columns
    # -----------------------------
    classification_columns = [
        "label",
        "conflict_related",
        "not_conflict_related",
        "pro_israel",
        "pro_palestine",
        "antisemite"
    ]
    for col in classification_columns:
        if col not in df.columns:
            df[col] = None

    # -----------------------------
    # 3. Helper: load existing backup (and deduplicate)
    # -----------------------------
    def load_backup(backup_csv):
        """
        Load existing backup CSV (if exists) into a DataFrame.
        Expects columns: [index, label, conflict_related, not_conflict_related, 
                          pro_israel, pro_palestine, antisemite].
        Returns a DataFrame with 'index' as a column (not as the DataFrame index).
        De-duplicates by "index" so we don't have multiple lines for the same row.
        """
        if backup_csv and os.path.exists(backup_csv):
            backup_df = pd.read_csv(backup_csv)
            if "index" not in backup_df.columns:
                print(f"[WARNING] '{backup_csv}' found but no 'index' column. Ignoring.")
                return pd.DataFrame(columns=["index"] + classification_columns)
            backup_df["index"] = backup_df["index"].astype(int)
            backup_df.drop_duplicates(subset="index", keep="last", inplace=True)
            return backup_df
        else:
            return pd.DataFrame(columns=["index"] + classification_columns)

    # -----------------------------
    # 4. Helper: flush new results to backup (overwriting with dedup)
    # -----------------------------
    def flush_buffer_to_backup(buffer, backup_csv):
        """
        Upsert approach:
          1) Load existing backup (if any).
          2) Concatenate 'buffer' with old backup.
          3) Drop duplicates on 'index'.
          4) Write out (overwrite) the backup CSV.
        """
        if not backup_csv or not buffer:
            return

        # 1. Load existing
        if os.path.exists(backup_csv):
            backup_df = pd.read_csv(backup_csv)
        else:
            backup_df = pd.DataFrame(columns=["index"] + classification_columns)

        # 2. Convert buffer -> DataFrame
        buffer_df = pd.DataFrame(buffer)

        # 3. Concatenate & drop duplicates
        combined_df = pd.concat([backup_df, buffer_df], ignore_index=True)
        combined_df.drop_duplicates(subset="index", keep="last", inplace=True)

        # 4. Overwrite the CSV
        combined_df.to_csv(backup_csv, index=False)

        # Clear the buffer
        buffer.clear()

    # -----------------------------
    # 5. Load backup & merge
    # -----------------------------
    existing_backup_df = load_backup(backup_file)

    if not existing_backup_df.empty:
        print(f"[INFO] Found backup with {len(existing_backup_df)} rows. Merging into main DataFrame.")

        # We'll merge the backup classifications into df by matching on the index.
        df.reset_index(drop=False, inplace=True)  # keep old index as a col named 'index'
        df.rename(columns={"index": "orig_index"}, inplace=True)

        backup_renamed = existing_backup_df.rename(columns={"index": "orig_index"})
        merged = pd.merge(
            df, 
            backup_renamed, 
            on="orig_index", 
            how="left", 
            suffixes=("", "_backup")
        )
        for col in classification_columns:
            merged[col] = merged[col].fillna(merged[f"{col}_backup"])
        backup_cols_to_drop = [f"{c}_backup" for c in classification_columns]
        merged.drop(columns=backup_cols_to_drop, inplace=True)
        df = merged
    else:
        print("[INFO] No valid backup found or file empty. Starting fresh.")
        if "orig_index" not in df.columns:
            df.reset_index(drop=False, inplace=True)
            df.rename(columns={"index": "orig_index"}, inplace=True)

    # -----------------------------
    # 6. Rate-limited batch classification function with exponential backoff
    # -----------------------------
    def is_rate_limit_error(exception):
        """Check if the exception is a rate limit error."""
        return isinstance(exception, Exception) and "Rate limit" in str(exception)
    
    @retry(
        retry=retry_if_exception_type((Exception,)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        before_sleep=lambda retry_state: print(f"[RETRY] Rate limit hit, retrying in {retry_state.next_action.sleep} seconds... (Attempt {retry_state.attempt_number}/5)")
    )
    @sleep_and_retry
    @limits(calls=300, period=60)  # Further reduced from 400 to 300 requests per 60 seconds to be even safer
    def openai_batch_label(texts_with_ids):
        """
        Calls OpenAI ChatCompletion for a batch of texts, returns labels.
        
        Args:
            texts_with_ids: List of tuples (row_idx, text)
            
        Returns:
            List of dicts with classification results
        """
        formatted_texts = ""
        for i, (_, text) in enumerate(texts_with_ids, 1):
            formatted_texts += f"Sample {i}: {text}\n\n"
            
        user_prompt = (
            "IMPORTANT: Classify each of the following protest descriptions ONLY into one of these exact categories:\n"
            "- 'pro-Israel' (for protests supporting Israel in the Israel-Palestine conflict)\n"
            "- 'pro-Palestinian' (for protests supporting Palestinians in the Israel-Palestine conflict)\n"
            "- 'pro-Both' (for protests related to the subject but not clearly pro one side)\n"
            "- 'unrelated' (for protests not related to the Israel-Palestine conflict)\n\n"
            "If any sample contains antisemitic content (Jew-hatred), append ' antisemite' to the label.\n\n"
            f"Input samples:\n{formatted_texts}\n"
            "Output format: Return a JSON array with one classification per sample, like this:\n"
            "[\"pro-Israel\", \"pro-Palestinian antisemite\", \"unrelated\", ...]\n"
            "DO NOT use any other labels besides these three (with optional ' antisemite' appended).\n"
            "Ensure the array length matches the number of input samples.\n"
            "Important: Return ONLY the JSON array with no markdown formatting, code blocks, or other text."
        )
        
        try:
            # Add jitter to prevent all workers hitting the API at exact same time
            jitter = random.uniform(0.1, 1.0)
            time.sleep(jitter)
            
            # Calculate approximate token count for rate limiting
            approx_token_count = len(user_prompt.split()) * 1.5  # rough estimate
            
            # If we're getting close to rate limit, add additional delay
            if approx_token_count > rate_limit_tpm / 10:
                delay = random.uniform(1.0, 5.0)
                print(f"[INFO] Large batch detected, adding {delay:.2f}s delay to avoid rate limits")
                time.sleep(delay)
                
            response = openai.ChatCompletion.create(
                model=model,
                temperature=0,
                messages=[{"role": "user", "content": user_prompt}],
            )
            content = response["choices"][0]["message"]["content"].strip()
            
            # Parse the JSON array response
            try:
                # Remove markdown code blocks if present (```json and ```)
                cleaned_content = content
                if cleaned_content.startswith("```"):
                    # Find the first and last ``` and remove them
                    first_end = cleaned_content.find("\n", 3)
                    if first_end != -1:
                        last_start = cleaned_content.rfind("```")
                        if last_start > first_end:
                            # Extract just the JSON content
                            cleaned_content = cleaned_content[first_end:last_start].strip()
                        else:
                            # Just remove the first marker
                            cleaned_content = cleaned_content[first_end:].strip()
                
                labels = json.loads(cleaned_content)
                if not isinstance(labels, list):
                    labels = [content] * len(texts_with_ids)  # Fallback if not a list
                
                # Ensure we have the right number of labels
                if len(labels) != len(texts_with_ids):
                    # Fill missing or trim extra
                    if len(labels) < len(texts_with_ids):
                        labels.extend(["unrelated"] * (len(texts_with_ids) - len(labels)))
                    else:
                        labels = labels[:len(texts_with_ids)]
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                print(f"[WARNING] Failed to parse JSON response: {content}")
                labels = [content] * len(texts_with_ids)
                
            results = []
            for (row_idx, _), label in zip(texts_with_ids, labels):
                label = label.strip() if isinstance(label, str) else "unrelated"
                conflict_related = int(("pro-Israel" in label) or ("pro-Palestinian" in label) or ("pro-Both" in label))
                antisemite = int("antisemite" in label)
                unrelated = int("unrelated" in label)
                
                results.append({
                    "index": row_idx,
                    "label": label,
                    "conflict_related": conflict_related,
                    "not_conflict_related": unrelated,
                    "pro_israel": int("pro-Israel" in label),
                    "pro_palestine": int("pro-Palestinian" in label),
                    "antisemite": antisemite,
                })
            return results
        except Exception as e:
            print(f"[ERROR] openai_batch_label: {e}")
            # Return default values for all texts in the batch
            return [
                {
                    "index": row_idx,
                    "label": None,
                    "conflict_related": 0,
                    "not_conflict_related": 0,
                    "pro_israel": 0,
                    "pro_palestine": 0,
                    "antisemite": 0,
                }
                for row_idx, _ in texts_with_ids
            ]

    # -----------------------------
    # 7. Worker function: classify a batch of rows
    # -----------------------------
    def classify_batch(batch_df):
        """
        Receives a batch of rows.
        - Skip rows that already have labels
        - Call OpenAI API for remaining rows
        - Returns a list of dicts with updated classification info
        """
        # Separate rows that need classification from those already classified
        need_classification = []
        already_classified = []

        for _, row in batch_df.iterrows():
            row_idx = row["orig_index"]
            label_exists = pd.notnull(row["label"]) and row["label"] != ""

            if label_exists:
                # Already classified from backup or prior run
                already_classified.append({
                    "index": row_idx,
                    "label": row["label"],
                    "conflict_related": row["conflict_related"],
                    "not_conflict_related": row["not_conflict_related"],
                    "pro_israel": row["pro_israel"],
                    "pro_palestine": row["pro_palestine"],
                    "antisemite": row["antisemite"],
                })
            else:
                need_classification.append((row_idx, row[text_column]))

        # If nothing needs classification, return already classified results
        if not need_classification:
            return already_classified

        # Classify the batch and combine with already classified results
        classified_results = openai_batch_label(need_classification)
        return already_classified + classified_results

    # -----------------------------
    # 8. Parallel classification with batching
    # -----------------------------
    total_rows = len(df)
    print(f"[INFO] Starting parallel classification on {total_rows} rows, max_workers={max_workers}, batch_size={batch_size}...")

    classification_buffer = []
    rows_processed_since_backup = 0

    # We'll chunk the dataset so that after each chunk, we can do a partial backup.
    num_chunks = ceil(total_rows / chunk_size)

    for chunk_idx in range(num_chunks):
        start_i = chunk_idx * chunk_size
        end_i = min((chunk_idx + 1) * chunk_size, total_rows)
        chunk_df = df.iloc[start_i:end_i]
        
        # Create batches within the chunk
        batches = []
        current_batch = []
        rows_to_process = 0
        
        for _, row in chunk_df.iterrows():
            # Check if row already has a label
            label_exists = pd.notnull(row["label"]) and row["label"] != ""
            
            current_batch.append(row)
            if not label_exists:
                rows_to_process += 1
                
            if len(current_batch) >= batch_size or rows_to_process >= batch_size:
                batches.append(pd.DataFrame(current_batch))
                current_batch = []
                rows_to_process = 0
                
        # Add any remaining rows
        if current_batch:
            batches.append(pd.DataFrame(current_batch))

        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(classify_batch, batch_df): i 
                for i, batch_df in enumerate(batches)
            }

            for future in as_completed(futures):
                try:
                    result_list = future.result()  # List of {"index": ..., "label": ..., ...}
                    classification_buffer.extend(result_list)
                except Exception as exc:
                    batch_idx = futures[future]
                    print(f"[ERROR] Batch {batch_idx} generated an exception: {exc}")

        # Update the main DataFrame with results
        for item in classification_buffer:
            i = df.index[df["orig_index"] == item["index"]]
            if len(i) == 1:  # Should be exactly 1
                i = i[0]
                df.at[i, "label"] = item["label"]
                df.at[i, "conflict_related"] = item["conflict_related"]
                df.at[i, "not_conflict_related"] = item["not_conflict_related"]
                df.at[i, "pro_israel"] = item["pro_israel"]
                df.at[i, "pro_palestine"] = item["pro_palestine"]
                df.at[i, "antisemite"] = item["antisemite"]
            else:
                print(f"[WARNING] Could not find unique row for orig_index={item['index']}")

        rows_processed_since_backup += len(classification_buffer)

        # Optionally flush partial progress to backup
        if backup_file and rows_processed_since_backup >= backup_every_n:
            flush_buffer_to_backup(classification_buffer, backup_file)
            rows_processed_since_backup = 0

        # Clear the buffer after writing to the DataFrame
        classification_buffer.clear()

    # -----------------------------
    # 9. Final flush (if desired)
    # -----------------------------
    if backup_file and classification_buffer:
        flush_buffer_to_backup(classification_buffer, backup_file)

    print("[INFO] Classification complete.")

    # Restore original index
    if "orig_index" in df.columns:
        df.set_index("orig_index", inplace=True)
        df.index.name = None

    return df

import time
# Example usage
if __name__ == "__main__":
    path = "<path to the data_subset to annotate>"
    toml_file = 'access_keys.toml'
    with open(toml_file, 'r') as f:
        toml_data = toml.load(f)
    openai_api_key = toml_data['keys']['open_AI_key']
    
    
    dfs = read_and_clean_data()
    dfs.keys()
    protests_to_annotate = pd.read_csv(path)
    
    t = time.time()
    protests_to_annotate = classify_protest_openai_parallel(protests_to_annotate.reset_index(drop=True), openai_api_key=openai_api_key, backup_file="protests_backup.csv", backup_every_n=500)
    protests_to_annotate.to_csv('protests_ranked.csv', index=False)
    print("Time taken: ", time.time() - t)
    
    
    

    
    # df = pd.read_csv(path)
    # df = df[df['conflict_related'] == 1].reset_index(drop=True)
    # df = classify_protest_openai_parallel(df, openai_api_key=openai_api_key, backup_file="riots_and_protests_labled_beckup.csv", backup_every_n=500)
    # df.to_csv('riots_and_protests_labled.csv', index=False)
