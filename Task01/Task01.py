import matplotlib.pyplot as plt

# Example continuous variable: Ages
ages = [18, 22, 21, 19, 24, 26, 29, 34, 40, 45, 50, 52, 54, 60, 62, 65, 70, 72, 75, 80]

# Example categorical variable: Gender counts
genders = {"Male": 55, "Female": 45, "Other": 5}

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Histogram for ages
axs[0].hist(ages, bins=10, edgecolor="black")
axs[0].set_title("Age Distribution")
axs[0].set_xlabel("Age")
axs[0].set_ylabel("Frequency")

# Bar chart for gender
axs[1].bar(genders.keys(), genders.values(), color=['blue', 'pink', 'green'])
axs[1].set_title("Gender Distribution")
axs[1].set_xlabel("Gender")
axs[1].set_ylabel("Count")

plt.tight_layout()
plt.show()
