import pandas

csv1 = pandas.read_csv('train_values.csv')
csv2 = pandas.read_csv('train_labels.csv')
merged = csv1.merge(csv2, on='building_id')
merged.to_csv("merged.csv", index=False)