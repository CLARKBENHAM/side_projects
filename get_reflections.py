import csv
with open('data/Work Summary  - When Distracted.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    count = 0
    for row in reader:
        if len(row) > 32 and row[32].strip() and row[32] != 'Productivity Value':
            print(f"{row[25]}: {row[32]}")
            count += 1
            if count >= 20:
                break
