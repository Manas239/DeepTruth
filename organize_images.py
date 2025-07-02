import os
import shutil

# Two possible source folders — pick one that actually exists
real_folder = "archive (6)/real_and_fake_face/training_real"
fake_folder = "archive (6)/real_and_fake_face/training_fake"

# If these don't exist, try this instead:
# real_folder = "archive (6)/real_and_fake_face_detection/real and fake face/training_real"
# fake_folder = "archive (6)/real_and_fake_face_detection/real and fake face/training_fake"

# Destination folders
real_dest = "data/real"
fake_dest = "data/fake"

# Make sure destination folders exist
os.makedirs(real_dest, exist_ok=True)
os.makedirs(fake_dest, exist_ok=True)

# Move real images
for file in os.listdir(real_folder):
    if file.lower().endswith(".jpg"):
        src = os.path.join(real_folder, file)
        shutil.move(src, os.path.join(real_dest, file))

# Move fake images
for file in os.listdir(fake_folder):
    if file.lower().endswith(".jpg"):
        src = os.path.join(fake_folder, file)
        shutil.move(src, os.path.join(fake_dest, file))

print("✅ Images moved to data/real and data/fake.")
