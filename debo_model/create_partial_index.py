import os
import pandas as pd
import glob

def main():
    print("Generating partial index from downloaded files...")
    
    # 1. Load the original CSV
    csv_path = "koi_cumulative_labeled.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    koi = pd.read_csv(csv_path, comment='#')
    
    # Create lookup map
    kepid_to_label = dict(zip(koi["kepid"], koi["signal_class"]))
    
    # 2. Find all downloaded .npz files
    npz_files = glob.glob("debo_model/datasets/*.npz")
    
    records = []
    for filepath in npz_files:
        filename = os.path.basename(filepath)
        kepid = int(filename.split(".")[0])
        
        label = kepid_to_label.get(kepid, "unknown")
        
        records.append({
            "kepid": kepid,
            "label": label,
            "file": filepath.replace("\\", "/") # standardize path separators
        })
        
    if not records:
        print("No downloaded .npz files found in debo_model/datasets.")
        return
        
    # 3. Save index
    df_index = pd.DataFrame(records)
    df_index.to_csv("debo_model/dataset_index.csv", index=False)
    print(f"Successfully generated debo_model/dataset_index.csv with {len(records)} records!")

if __name__ == "__main__":
    main()
