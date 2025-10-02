import os
import sys
import random
import math
import numpy as np
import pandas as pd

def normalize(x):
    x = np.array(x)
    return ((x - np.mean(x))/(np.max(x)-np.min(x))).tolist()

def rotate_point(x, y, angle):
    x1 = x * math.cos(angle) - y * math.sin(angle)
    y1 = x * math.sin(angle) + y * math.cos(angle)
    return round(x1), round(y1)

def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        header = lines[0]  # Save the header line
        new_lines = [header]  # Start the new file with the same header
        angle = random.uniform(-math.pi/6, math.pi/6)
        for i, line in enumerate(lines):
            if i == 0:
                continue
            vals = [eval(i) for i in line.split()]
            x, y = vals[:2]
            x1, y1 = rotate_point(x, y, angle)
            vals[0] = x1
            vals[1] = y1
            string_array = [str(x) for x in vals]
            result = " ".join(string_array)
            new_lines.append(result + "\n")
        
        folder_path, file_name = os.path.split(file_path)
        new_folder_path = os.path.join(folder_path, "rot")
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
        new_file_path = os.path.join(new_folder_path, file_name.replace(".txt", "_rot.txt"))
        with open(new_file_path, 'w') as new_file:
            new_file.writelines(new_lines)

def main():
    folder_path = sys.argv[1]
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    chosen_files = random.sample(files, int(len(files)*0.1))

    for file_name in chosen_files:
        file_path = os.path.join(folder_path, file_name)
        process_file(file_path)

if __name__ == "__main__":
    main()
