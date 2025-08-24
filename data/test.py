import csv

# File paths
paths_file = 'output.txt'  # File with paths
labels_file = 'nabraj1.txt'  # File with labels
csv_file = 'output.csv'  # The final CSV file

# Read paths from output.txt
with open(paths_file, 'r', encoding='utf-8') as f:
    paths = [line.strip() for line in f]

# Read labels from nabraj.txt
with open(labels_file, 'r', encoding='utf-8') as f:
    labels = [line.strip() for line in f]

# Check if both files have the same number of lines
if len(paths) != len(labels):
    print("The number of paths and labels do not match!")
else:
    # Prepare data by combining paths and labels
    data = zip(paths, labels)

    # Write data to CSV, either appending or creating a new file
    with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header only if the file is empty (new file)
        if csvfile.tell() == 0:
            writer.writerow(['Path', 'Label'])

        # Write the combined data (path, label)
        writer.writerows(data)

    print(f"Data has been written to {csv_file}")
