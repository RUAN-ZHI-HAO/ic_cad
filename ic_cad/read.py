import json

with open("equiv_groups.json", "r", encoding="utf-8") as f:
    parsed = json.load(f)
    # print(parsed)

total_cell_count = 0
for group in parsed:
    total_cell_count += len(group)

print(f"Total cell count: {total_cell_count}")
