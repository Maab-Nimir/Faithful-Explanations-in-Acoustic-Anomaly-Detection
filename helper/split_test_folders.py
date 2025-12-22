import os
import shutil

root_dir = "/home/ulaval.ca/maelr5/scratch/acoustic-monitoring/dcase2022/dev_spectrograms"

def split_test_sets(root_dir):
    for machine in os.listdir(root_dir):
        machine_path = os.path.join(root_dir, machine)
        if not os.path.isdir(machine_path):
            continue

        test_path = os.path.join(machine_path, "test")
        if not os.path.exists(test_path):
            continue

        print(f"Processing {machine}...")

        normal_dir = os.path.join(test_path, "normal")
        anomaly_dir = os.path.join(test_path, "anomaly")
        os.makedirs(normal_dir, exist_ok=True)
        os.makedirs(anomaly_dir, exist_ok=True)

        # Move test files
        for file in os.listdir(test_path):
            file_path = os.path.join(test_path, file)
            if not os.path.isfile(file_path):
                continue

            # Classify based on filename
            if "normal" in file.lower():
                dest = os.path.join(normal_dir, file)
            elif "anomaly" in file.lower():
                dest = os.path.join(anomaly_dir, file)
            else:
                print(f"⚠️ Skipping unrecognized file: {file}")
                continue

            shutil.move(file_path, dest)

        print(f"✅ {machine}: moved files to test/normal and test/anomaly.\n")

if __name__ == "__main__":
    split_test_sets(root_dir)
